"""Thin helpers around the diffusers FLUX 3 Image Klein modular pipeline for the MPS notebooks."""

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import ModularPipeline
from PIL import Image

from mps_flux_3_fp8.utils.log import logger

REPO_DISTILLED = "black-forest-labs/flux-3-image-klein"
REPO_BASE = "black-forest-labs/flux-3-image-klein-base"
GiB = 1024**3


@dataclass(frozen=True)
class MpsMemory:
    """MPS memory snapshot in GiB: tensors currently allocated and the total held by the Metal driver."""

    allocated: float
    driver: float

    def __str__(self) -> str:
        return f"allocated={self.allocated:.2f} GiB, driver={self.driver:.2f} GiB"


@dataclass(frozen=True)
class RunResult:
    """One timed generation: the image, the wall-clock seconds and the MPS memory right after it."""

    image: Image.Image
    seconds: float
    memory: MpsMemory


def mps_memory() -> MpsMemory:
    """Return the current MPS memory usage.

    Returns:
        MpsMemory: Allocated and driver memory in GiB.
    """
    return MpsMemory(torch.mps.current_allocated_memory() / GiB, torch.mps.driver_allocated_memory() / GiB)


def free_mps() -> MpsMemory:
    """Run the GC and release cached MPS blocks.

    Returns:
        MpsMemory: Memory after the release.
    """
    gc.collect()
    torch.mps.empty_cache()
    return mps_memory()


def module_gib(module: torch.nn.Module) -> float:
    """Size of a module's parameters and buffers in GiB (FP8 weights count one byte per element).

    Args:
        module (torch.nn.Module): The module to measure.

    Returns:
        float: Size in GiB.
    """
    tensors = [*module.parameters(), *module.buffers()]
    return sum(t.numel() * t.element_size() for t in tensors) / GiB


def load_pipeline(repo: str = REPO_DISTILLED, with_upsampler: bool = True, device: str = "mps") -> ModularPipeline:
    """Build the FLUX 3 Klein modular pipeline, load all its weights in bf16 (VAE in fp32) and move it to `device`.

    Args:
        repo (str): Hugging Face repo; the distilled one resolves to `Flux3DistilledModularPipeline` (16 steps, no CFG).
        with_upsampler (bool): Keep the 9B prompt-upsampler step; when False, `prompt` must already be a full caption.
        device (str): Target device.

    Returns:
        ModularPipeline: The ready pipeline.
    """
    blocks = ModularPipeline.from_pretrained(repo).blocks
    if not with_upsampler:
        blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline(repo)
    pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
    pipe.to(device)
    logger.info("Loaded %s (upsampler=%s) on %s: %s", repo, with_upsampler, device, mps_memory())
    return pipe


def load_upsampler(repo: str = REPO_DISTILLED, device: str = "mps") -> ModularPipeline:
    """Build the prompt-upsampler step alone (black-forest-labs/flux-3-image-klein-pup, Qwen3.5-9B, bf16).

    Args:
        repo (str): The FLUX 3 repo whose `modular_model_index.json` points at the upsampler weights.
        device (str): Target device.

    Returns:
        ModularPipeline: A pipeline whose call with `output="prompt"` returns the expanded caption.
    """
    blocks = ModularPipeline.from_pretrained(repo).blocks
    pup = blocks.sub_blocks.pop("prompt_upsampler").init_pipeline(repo)
    pup.load_components(dtype=torch.bfloat16)
    pup.to(device)
    return pup


def generate(
    pipe: ModularPipeline,
    prompt: str,
    seed: int = 0,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int | None = None,
) -> RunResult:
    """Run one text-to-image generation with CPU-seeded noise (identical noise across devices and dtypes).

    Args:
        pipe (ModularPipeline): A loaded FLUX 3 pipeline.
        prompt (str): The request (or the full caption, for a pipeline without the upsampler).
        seed (int): Noise seed.
        height (int): Output height in pixels, multiple of 16.
        width (int): Output width in pixels, multiple of 16.
        num_inference_steps (int | None): Override the pipeline default (16 for distilled, 50 for base).

    Returns:
        RunResult: The image, elapsed seconds and MPS memory after the run.
    """
    assert height % 16 == 0 and width % 16 == 0, "height and width must be multiples of 16"
    kwargs = {} if num_inference_steps is None else {"num_inference_steps": num_inference_steps}
    torch.mps.synchronize()
    start = time.perf_counter()
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        generator=torch.Generator().manual_seed(seed),
        output="images",
        **kwargs,
    )[0]
    torch.mps.synchronize()
    seconds = time.perf_counter() - start
    result = RunResult(image, seconds, mps_memory())
    logger.info("Generated %dx%d in %.1fs (%s)", width, height, seconds, result.memory)
    return result


def psnr(a: Image.Image, b: Image.Image) -> float:
    """Peak signal-to-noise ratio between two same-size RGB images, in dB (inf when identical).

    Args:
        a (Image.Image): Reference image.
        b (Image.Image): Test image.

    Returns:
        float: PSNR in dB.
    """
    assert a.size == b.size, f"size mismatch {a.size} vs {b.size}"
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def save(image: Image.Image, path: str | Path) -> Path:
    """Save an image, creating parent directories.

    Args:
        image (Image.Image): Image to save.
        path (str | Path): Destination file.

    Returns:
        Path: The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path

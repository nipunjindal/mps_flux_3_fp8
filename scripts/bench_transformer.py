#!/usr/bin/env python3
"""Benchmark one FLUX 3 Klein transformer step on MPS: bf16 vs the FP8r backends (torch >= 2.15 for FP8).

Uses the real distilled architecture (5.6B params, flow_map=True) with random weights, so it runs without access to
the gated checkpoint. Speed and memory match the real model; outputs are only compared against the bf16 run.

    .venv-torch215/bin/python scripts/bench_transformer.py --size 1024 --text-tokens 512
"""

import argparse
import sys
import time

import torch
from diffusers import Flux3Transformer2DModel

from mps_flux_3_fp8.flux3 import free_mps, module_gib
from mps_flux_3_fp8.fp8 import probe_mps_fp8, quantize_linears_fp8r, set_fp8r_backend
from mps_flux_3_fp8.utils.log import logger


def make_inputs(size: int, text_tokens: int) -> dict[str, torch.Tensor]:
    """Random step inputs for a `size` x `size` image; shapes follow the pipeline (16x16 pixels per token)."""
    side = size // 16
    generator = torch.Generator().manual_seed(0)
    h, w = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    img_ids = torch.stack([torch.zeros_like(h), h, w, torch.zeros_like(h)], -1).reshape(-1, 4)
    txt_ids = torch.zeros(text_tokens, 4, dtype=torch.long)
    txt_ids[:, 3] = torch.arange(text_tokens)
    tensors = {
        "hidden_states": torch.randn(1, side * side, 128, generator=generator),
        "encoder_hidden_states": torch.randn(1, text_tokens, 20480, generator=generator),
        "timestep": torch.tensor([0.75]),
        "target_timestep": torch.tensor([0.70]),
        "img_ids": img_ids,
        "txt_ids": txt_ids,
    }
    return {k: v.to("mps", torch.bfloat16 if v.is_floating_point() else v.dtype) for k, v in tensors.items()}


def time_step(
    model: Flux3Transformer2DModel, inputs: dict[str, torch.Tensor], iters: int
) -> tuple[float, torch.Tensor]:
    """Mean seconds per forward (after one warm-up) and the output of the last call."""
    with torch.inference_mode():
        out = model(**inputs, return_dict=False)[0]
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            out = model(**inputs, return_dict=False)[0]
        torch.mps.synchronize()
    return (time.perf_counter() - start) / iters, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=1024, help="image side in pixels")
    parser.add_argument("--text-tokens", type=int, default=512)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    support = probe_mps_fp8()
    logger.info("torch %s, MPS FP8 support: %s", torch.__version__, support)

    torch.manual_seed(0)
    with torch.device("mps"):
        model = Flux3Transformer2DModel(flow_map=True).to(torch.bfloat16).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=module.in_features**-0.5)
    inputs = make_inputs(args.size, args.text_tokens)

    seconds, reference = time_step(model, inputs, args.iters)
    rows = [("bf16", module_gib(model), seconds, 0.0)]
    if support.float8_e4m3fn:
        quantize_linears_fp8r(model)
        free_mps()
        gib = module_gib(model)
        backends = ["diffusers", "dequant", "scaled_mm"] if support.scaled_mm else ["diffusers", "dequant"]
        for backend in backends:
            set_fp8r_backend(model, backend)
            seconds, out = time_step(model, inputs, args.iters)
            err = ((out.float() - reference.float()).norm() / reference.float().norm()).item()
            rows.append((f"fp8r/{backend}", gib, seconds, err))

    print(f"\n{args.size}x{args.size}, {args.text_tokens} text tokens, torch {torch.__version__}")
    print(f"{'variant':<18}{'weights GiB':>12}{'s/step':>9}{'16 steps':>10}{'rel err':>9}")
    for name, gib, sec, err in rows:
        print(f"{name:<18}{gib:>12.2f}{sec:>9.2f}{sec * 16:>9.1f}s{err:>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

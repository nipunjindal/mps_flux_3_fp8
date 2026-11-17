"""FP8 (E4M3, per-row scaled) linear layers for FLUX 3 on Apple Silicon (MPS).

torch 2.15 is the first release whose MPS backend has a `float8_e4m3fn` dtype and an `F.scaled_mm` kernel; on
torch 2.14 creating an FP8 tensor on MPS fails with "Undefined type Float8_e4m3fn".

The FP8 weights live in diffusers' `BFLFP8RLinear` (weight `[N, K]` E4M3, fp32 per-row scale `[N]`), which is what
`Flux3Transformer2DModel.from_single_file(..., quantization_config=BFLQuantizationConfig())` builds from the
official `variants/fp8r` checkpoint, or what `quantize_linears_fp8r` builds from a bf16 model. Off-CUDA its forward
dequantizes the weight through fp32 on every call; `set_fp8r_backend` swaps in one of the MPS forwards below.
"""

import platform
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from diffusers.quantizers.bfl.utils import FP8_MAX, BFLFP8RLinear, quantize_fp8r
from torch import nn

from mps_flux_3_fp8.utils.log import logger

Fp8rBackend = Literal["diffusers", "dequant", "scaled_mm"]
"""How a `BFLFP8RLinear` runs its matmul.

- `diffusers`: the stock forward, weight dequantized through fp32 to the compute dtype, then a bf16 GEMM.
- `dequant`: weight-only FP8; the E4M3 weight is cast straight to the compute dtype and the per-row scale is applied
  to whichever of the weight `[N, K]` or the output `[M, N]` is smaller. Fastest on MPS.
- `scaled_mm`: true FP8 compute; activations are quantized per row and the GEMM runs in `F.scaled_mm` (RowWise).
"""

# Linears inside these submodules hold ~99% of the transformer weights; embedders, modulations and proj_out stay bf16.
FLUX3_QUANTIZED_PREFIXES = ("mode_blocks.", "txt_mode_blocks.", "single_transformer_blocks.")


@dataclass(frozen=True)
class MpsFp8Support:
    """What the current torch build supports for FP8 on MPS."""

    torch_version: str
    mps_available: bool
    float8_e4m3fn: bool
    scaled_mm: bool
    error: str | None = None


def probe_mps_fp8() -> MpsFp8Support:
    """Check whether the MPS backend can store `float8_e4m3fn` tensors and run `F.scaled_mm` on them.

    Returns:
        MpsFp8Support: The probe result; `error` carries the first failure message, if any.
    """
    version = torch.__version__
    if not torch.backends.mps.is_available():
        return MpsFp8Support(version, False, False, False, f"MPS not available on {platform.platform()}")

    x = torch.ones(16, 16, device="mps")
    try:
        x8 = x.to(torch.float8_e4m3fn)
    except (RuntimeError, TypeError) as e:
        return MpsFp8Support(version, True, False, False, str(e))

    scale = torch.ones(16, 1, device="mps")
    try:
        out = F.scaled_mm(x8, x8.t(), scale, F.ScalingType.RowWise, scale.t(), F.ScalingType.RowWise)
    except (AttributeError, RuntimeError, NotImplementedError, TypeError) as e:
        return MpsFp8Support(version, True, True, False, str(e))
    assert torch.allclose(out.float().cpu(), torch.full((16, 16), 16.0)), "F.scaled_mm gave a wrong result"
    return MpsFp8Support(version, True, True, True)


def fp8r_linear_dequant(module: BFLFP8RLinear, inputs: torch.Tensor) -> torch.Tensor:
    """Weight-only FP8 forward: cast E4M3 -> compute dtype, scale the smaller of weight or output, bf16 GEMM.

    Args:
        module (BFLFP8RLinear): Holds `weight` `[N, K]` E4M3 and `weight_scale` `[N]` fp32.
        inputs (torch.Tensor): Activations `[..., K]`.

    Returns:
        torch.Tensor: `[..., N]` in `module.compute_dtype`.
    """
    dtype = module.compute_dtype
    x = inputs.to(dtype)
    scale = module.weight_scale.to(dtype)
    rows = x.numel() // module.in_features
    if rows <= module.in_features:
        output = F.linear(x, module.weight.to(dtype)) * scale
    else:
        output = F.linear(x, module.weight.to(dtype) * scale[:, None])
    return output if module.bias is None else output + module.bias


def fp8r_linear_scaled_mm(module: BFLFP8RLinear, inputs: torch.Tensor) -> torch.Tensor:
    """True FP8 forward: per-row E4M3 activations x per-row E4M3 weights through `F.scaled_mm` (torch >= 2.15).

    Args:
        module (BFLFP8RLinear): Holds `weight` `[N, K]` E4M3 and `weight_scale` `[N]` fp32.
        inputs (torch.Tensor): Activations `[..., K]`.

    Returns:
        torch.Tensor: `[..., N]` in `module.compute_dtype`.
    """
    x = inputs.reshape(-1, module.in_features).float()
    x_scale = (x.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    x_q = (x / x_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    output = F.scaled_mm(
        x_q,
        module.weight.t(),
        x_scale,
        F.ScalingType.RowWise,
        module.weight_scale[None, :],
        F.ScalingType.RowWise,
        output_dtype=torch.bfloat16,
    )
    output = output.reshape(*inputs.shape[:-1], module.out_features).to(module.compute_dtype)
    return output if module.bias is None else output + module.bias


FORWARDS: dict[Fp8rBackend, Callable[[BFLFP8RLinear, torch.Tensor], torch.Tensor] | None] = {
    "diffusers": None,
    "dequant": fp8r_linear_dequant,
    "scaled_mm": fp8r_linear_scaled_mm,
}


def set_fp8r_backend(model: nn.Module, backend: Fp8rBackend) -> int:
    """Point every `BFLFP8RLinear` in `model` at the given forward implementation.

    Args:
        model (nn.Module): A model containing `BFLFP8RLinear` layers.
        backend (Fp8rBackend): The forward to use; `diffusers` restores the stock one.

    Returns:
        int: Number of layers switched.

    Raises:
        AssertionError: If `backend` is unknown.
    """
    assert backend in FORWARDS, f"unknown backend {backend!r}"
    forward = FORWARDS[backend]
    layers = [m for m in model.modules() if isinstance(m, BFLFP8RLinear)]
    for layer in layers:
        layer.__dict__.pop("forward", None)
        if forward is not None:
            layer.forward = partial(forward, layer)
    logger.info("FP8r backend %s set on %d layers", backend, len(layers))
    return len(layers)


def quantize_linears_fp8r(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    prefixes: tuple[str, ...] = FLUX3_QUANTIZED_PREFIXES,
) -> int:
    """Replace `nn.Linear` layers under `prefixes` with `BFLFP8RLinear` holding E4M3 weights + fp32 per-row scales.

    Same layout as the official `variants/fp8r` checkpoint, so `set_fp8r_backend` applies to either. Quantization
    runs on the weight's current device and frees the bf16 weight.

    Args:
        model (nn.Module): The model to quantize in place.
        compute_dtype (torch.dtype): Activation / output dtype of the new layers.
        prefixes (tuple[str, ...]): Qualified-name prefixes of the linears to convert; `("",)` converts all.

    Returns:
        int: Number of layers converted.
    """
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if type(module) is nn.Linear and name.startswith(prefixes)
    ]
    for name, linear in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        weight_q, weight_scale = quantize_fp8r(linear.weight.data)
        device = linear.weight.device
        layer = BFLFP8RLinear(
            linear.in_features, linear.out_features, linear.bias is not None, compute_dtype, device="meta"
        )
        layer.weight = nn.Parameter(weight_q, requires_grad=False)
        layer.weight_scale = weight_scale.to(device)
        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.data.to(compute_dtype), requires_grad=False)
        setattr(parent, child, layer)
    logger.info("Quantized %d linears to fp8r", len(targets))
    return len(targets)

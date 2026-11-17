"""Sweep the token count M for one FLUX-sized linear (K=3072, N=12288) on MPS: bf16 vs FP8 paths (torch >= 2.15).

Prints CSV with ms per call for: bf16 GEMM, FP8 weight dequant + bf16 GEMM, `F.scaled_mm` with pre-quantized
activations, and `F.scaled_mm` with per-call (dynamic) activation quantization.
"""

import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

DEVICE = "mps"
K, N = 3072, 12288
TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4608)
FP8_MAX = 448.0
ROW = F.ScalingType.RowWise


def time_ms(fn: Callable[[], torch.Tensor], iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def dynamic_scaled_mm(x: torch.Tensor, w8t: torch.Tensor, scale_w: torch.Tensor) -> torch.Tensor:
    scale_x = (x.abs().amax(1, keepdim=True).float() / FP8_MAX).clamp(min=1e-12)
    x8 = (x / scale_x).to(torch.float8_e4m3fn)
    return F.scaled_mm(x8, w8t, scale_x, ROW, scale_w, ROW, output_dtype=torch.bfloat16)


def main() -> None:
    w = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) / K**0.5
    w8t = w.to(torch.float8_e4m3fn).t()
    scale_w = torch.ones(1, N, device=DEVICE)
    print("M,bf16_ms,fp8_dequant_ms,fp8_scaled_mm_prequant_ms,fp8_scaled_mm_dynamic_ms")
    for m in TOKENS:
        x = torch.randn(m, K, device=DEVICE, dtype=torch.bfloat16)
        x8 = x.to(torch.float8_e4m3fn)
        scale_x = torch.ones(m, 1, device=DEVICE)
        times = (
            time_ms(lambda x=x: x @ w.t()),
            time_ms(lambda x=x: x @ w8t.to(torch.bfloat16)),
            time_ms(lambda x8=x8, s=scale_x: F.scaled_mm(x8, w8t, s, ROW, scale_w, ROW, output_dtype=torch.bfloat16)),
            time_ms(lambda x=x: dynamic_scaled_mm(x, w8t, scale_w)),
        )
        print(f"{m}," + ",".join(f"{t:.3f}" for t in times))


if __name__ == "__main__":
    main()

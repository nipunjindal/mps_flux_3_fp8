# mps_flux_3_fp8

FLUX 3 Image Klein with **diffusers 0.43** on Apple Silicon (MPS): a bf16 baseline on torch 2.14, then FP8 on
**torch 2.15**, the first release with `float8_e4m3fn` and `F.scaled_mm` on MPS.

## Setup

```bash
make kernels   # builds .venv-torch214 + .venv-torch215 and registers Jupyter kernels for both
```

The model repos are gated: accept the license at https://huggingface.co/black-forest-labs/flux-3-image-klein and
export `HF_TOKEN` (or `hf auth login`). The prompt upsampler (`flux-3-image-klein-pup`, 18.8 GB) is public.

## Notebooks

| notebook | kernel | does |
|---|---|---|
| `notebooks/01_flux3_bf16_mps.ipynb` | FLUX3 (torch 2.14) | prompt upsampling → `outputs/caption.txt`, bf16 generation |
| `notebooks/02_flux3_fp8_mps.ipynb` | FLUX3 (torch 2.15) | bf16 vs FP8 (fp8r) with three MPS matmul backends, same caption + seed |

Outputs (images, `results.json`) go to `outputs/`.

## FP8 on MPS (torch 2.15)

`src/mps_flux_3_fp8/fp8.py` runs diffusers' `BFLFP8RLinear` (E4M3 weights, fp32 per-row scales) with:

- `diffusers` — stock forward: dequantize through fp32 each call, bf16 GEMM
- `dequant` — fp8→bf16 cast, scale applied to the smaller of weight/output, bf16 GEMM
- `scaled_mm` — true FP8 compute through `F.scaled_mm` (RowWise)

`make bench` (random weights, real architecture, 1024², 512 text tokens, M5 Max):

| variant | transformer weights | s/step | 16 steps |
|---|---|---|---|
| bf16 (torch 2.14 / 2.15) | 10.34 GiB | 1.10 / 1.09 | 17.6 s |
| fp8r / diffusers | 5.43 GiB | 1.38 | 22.2 s |
| fp8r / dequant | 5.43 GiB | 1.32 | 21.1 s |
| fp8r / scaled_mm | 5.43 GiB | 6.11 | 97.8 s |

End-to-end (notebooks, official `variants/fp8r` checkpoint, 1024², 16 steps, upsampled caption, seed 0, M5 Max):

| run | torch | seconds | transformer | MPS allocated | PSNR vs bf16 |
|---|---|---|---|---|---|
| bf16 | 2.14.1 | 18.0 | 10.34 GiB | 18.98 GiB | — |
| bf16 | 2.15.0 | 17.9 | 10.34 GiB | 18.98 GiB | — |
| fp8r / diffusers | 2.15.0 | 20.0 | 5.84 GiB | 14.48 GiB | 23.3 dB |
| fp8r / dequant | 2.15.0 | 20.8 | 5.84 GiB | 14.48 GiB | 21.5 dB |
| fp8r / scaled_mm | 2.15.0 | 96.7 | 5.84 GiB | 14.48 GiB | 22.1 dB |

On MPS today FP8 is a memory win (weights halved), not a speed win: the `F.scaled_mm` kernel is ~9× slower than a
bf16 GEMM, so weight-only FP8 is the practical mode.

## What we learned

All measured on M5 Max, macOS 26.7.1, torch 2.15.0, diffusers 0.43.0.

### 1. "FP8 math" on MPS is FP8 storage, fp16 math

torch 2.15's MPS `F.scaled_mm`
([`ScaledMM.metal`](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/native/mps/kernels/ScaledMM.metal))
loads FP8 elements one at a time, converts them to `half` in 32×32 threadgroup tiles, and multiplies with
`simdgroup_half8x8` (or MPP `matmul2d` on M5). Scales are applied in the epilogue. There is no FP8 arithmetic,
and the generic tiled kernel reaches ~7 TFLOPS vs 50–63 TFLOPS for Apple's bf16/fp16 GEMM. Native FP8 tensors in
Metal TensorOps (`MTLTensorDataTypeMetalFloat8E4M3`) arrive in macOS 27
([WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/)).

### 2. FP8 wins only below ~8 tokens; FLUX runs at 4,608

`scripts/sweep_fp8_gemm.py`, one FLUX linear (K=3072, N=12288), ms per call:

| tokens M | bf16 | fp8 dequant + bf16 | `scaled_mm` (pre-quantized x) | `scaled_mm` (quantize x per call) |
|---|---|---|---|---|
| 1 | 0.193 | 0.569 | **0.052** | **0.079** |
| 4 | 0.159 | 0.497 | **0.109** | **0.138** |
| 8 | 0.182 | 0.465 | 0.394 | 0.398 |
| 64 | 0.175 | 0.525 | 0.736 | 0.775 |
| 512 | 0.661 | 0.914 | 5.375 | 5.329 |
| 4608 (FLUX 1024²) | 5.554 | 5.714 | 47.055 | 47.486 |

- For M ≤ 4, torch dispatches a separate `scaled_mm_few_rows` kernel that streams FP8 weights. It is up to 3.7×
  faster than bf16, which is the LLM-decode regime.
- From M = 8 the generic tiled kernel takes over, and it is already slower than bf16.
- A 1024² FLUX image has 4,096 image + 512 text tokens, so each GEMM sits deep in the compute-bound regime.
  Reading 10 GiB of weights per step costs ~20 ms of a ~1.1 s step, so halving them can save at most ~2%.
- Dequant-then-GEMM costs a fixed ~0.3 ms per call, the full weight cast. That is negligible at FLUX's M but it
  dominates small M.

### 3. BFL's official fp8r checkpoint keeps the text stream in bf16

The checkpoint (`variants/fp8r`) has 234 E4M3 linears with per-row fp32 scales. Every image, `image_cond` and
single-stream projection is FP8, and so are the 6 shared modulation linears. All 30 `txt_mode_blocks` linears stay
bf16, along with the embedders, `time_in` and the final layer. A naive "quantize every block" pass (this repo's
`quantize_linears_fp8r`) converts 258 layers instead: it adds the text blocks and leaves the modulations out.

### 4. Pixel PSNR is a poor FP8 quality metric for distilled FLUX

Pairwise PSNR of the five outputs (same caption, seed and steps):

| pair | what differs | PSNR |
|---|---|---|
| bf16 torch 2.14 vs bf16 torch 2.15 | only torch kernels | 29.3 dB |
| fp8 diffusers vs dequant vs scaled_mm | same FP8 weights, different matmul order | 24.5–25.6 dB |
| bf16 vs any FP8 | weights quantized | 21.5–23.7 dB |

Changing only the matmul kernel, with identical FP8 weights, already moves images by ~25 dB. The 16-step distilled
sampler amplifies tiny numeric differences into fine-detail changes, and all five images are visually the same
scene. Judge FP8 quality with perceptual or prompt-alignment metrics, not pixel PSNR against one bf16 run.

### 5. Attention is already a fast fused kernel

diffusers' default `native` backend calls `F.scaled_dot_product_attention`. On MPS that dispatches to torch's tiled
`sdpa_prefill_mps` kernel (MPP on M5) for head_dim 128 without a mask: ~5.3 ms per call (≈50 TFLOPS) at
24 heads × 4,608 tokens. With `PYTORCH_MPS_DISABLE_PREFILL_ATTENTION=1` it falls back to MPSGraph at 47 ms (9×
slower). `flash_attn`, xformers and sage are not used and don't support MPS. Attention is ~16% of a step; GEMMs
dominate.

### 6. fp16 GEMM can beat bf16 on M5

At 4608×3072×3072, fp16 runs at 62 TFLOPS vs 50 for bf16. Larger shapes are on par (~63). That gap is bigger than
anything FP8 currently offers for FLUX on MPS.

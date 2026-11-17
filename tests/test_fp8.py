import pytest
import torch
from diffusers.quantizers.bfl.utils import BFLFP8RLinear
from torch import nn

from mps_flux_3_fp8.fp8 import probe_mps_fp8, quantize_linears_fp8r, set_fp8r_backend

SUPPORT = probe_mps_fp8()
requires_mps_fp8 = pytest.mark.skipif(not SUPPORT.scaled_mm, reason=f"no MPS FP8: {SUPPORT.error}")


class TinyModel(nn.Module):
    """Mimics FLUX 3 naming: linears under `single_transformer_blocks` get quantized, `proj_out` stays bf16."""

    def __init__(self) -> None:
        super().__init__()
        self.single_transformer_blocks = nn.ModuleList([nn.Linear(256, 512, bias=False), nn.Linear(512, 256)])
        self.proj_out = nn.Linear(256, 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.single_transformer_blocks:
            x = layer(x)
        return self.proj_out(x)


def make_model() -> TinyModel:
    torch.manual_seed(0)
    return TinyModel().to("mps", torch.bfloat16)


def forward(model: TinyModel, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@requires_mps_fp8
def test_quantize_only_targets_prefixed_linears():
    model = make_model()
    assert quantize_linears_fp8r(model) == 2
    assert all(isinstance(m, BFLFP8RLinear) for m in model.single_transformer_blocks)
    assert type(model.proj_out) is nn.Linear
    assert model.single_transformer_blocks[0].weight.dtype == torch.float8_e4m3fn


@requires_mps_fp8
@pytest.mark.parametrize("rows", [8, 1024])
@pytest.mark.parametrize("backend", ["diffusers", "dequant", "scaled_mm"])
def test_backends_match_bf16(backend, rows):
    model = make_model()
    x = torch.randn(2, rows // 2, 256, device="mps", dtype=torch.bfloat16)
    reference = forward(model, x)
    quantize_linears_fp8r(model)
    assert set_fp8r_backend(model, backend) == 2
    out = forward(model, x)
    assert out.shape == reference.shape and out.dtype == torch.bfloat16
    assert rel_err(out, reference) < 0.08


@requires_mps_fp8
def test_backend_switch_is_reversible():
    model = make_model()
    quantize_linears_fp8r(model)
    x = torch.randn(4, 256, device="mps", dtype=torch.bfloat16)
    stock = forward(model, x)
    set_fp8r_backend(model, "scaled_mm")
    set_fp8r_backend(model, "diffusers")
    assert torch.equal(forward(model, x), stock)

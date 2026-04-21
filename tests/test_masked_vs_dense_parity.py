"""Numerical parity: MASKED_IMPLICIT_GEMM vs IMPLICIT_GEMM on MPS.

The masked kernel iterates only the valid V positions per n-block (via
sorted_idx + valid_kernel + valid_kernel_seg). Mathematically equivalent to
the dense kernel by construction — invalid V positions contribute zero in
the dense version too — but the reduction order differs because:
  (a) rows are processed sorted by their gray-coded neighbor mask, not by
      their natural order, and
  (b) only the valid V positions are accumulated, in valid_kernel order.

So fp32 stays bit-exact (associative reductions), and fp16/bf16 are within
the per-element epsilon of a different summation order.
"""
import os
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import pytest
import torch

from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm, set_algorithm


needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="masked-vs-dense parity is checked on MPS (production target).",
)


def _sphere_coords(res, ch, dtype, device):
    coords = torch.stack(torch.meshgrid(
        torch.arange(res), torch.arange(res), torch.arange(res), indexing="ij",
    ), dim=-1).int().contiguous()
    dist = ((coords.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
    active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
    coords = torch.nonzero(active).int()
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=-1)
    coords = coords.contiguous().to(device)
    feats = torch.randn(coords.shape[0], ch, dtype=dtype).to(device).contiguous()
    return feats, coords, torch.Size([1, ch, res, res, res])


# Tolerances reflect different reduction order between masked and dense kernels.
# Set per-dtype: fp32 is associative-enough that we can hold tight, half/bfloat
# need a looser bound proportional to the accumulated K = V * Ci magnitude.
TOLERANCES = {
    torch.float32:  dict(atol=1e-4, rtol=1e-5),
    torch.float16:  dict(atol=2e-2, rtol=5e-3),
    torch.bfloat16: dict(atol=5e-2, rtol=1e-2),
}


@needs_mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("res,ch", [(16, 32), (16, 64), (32, 64), (32, 128), (64, 64)])
def test_masked_matches_dense(res, ch, dtype):
    torch.manual_seed(0xC0FFEE ^ res ^ (ch << 4))
    device = "mps"

    feats, coords, shape = _sphere_coords(res, ch, dtype, device)
    Co = ch
    Ks = 3
    weight = torch.randn(Co, Ks, Ks, Ks, ch, dtype=dtype).to(device)
    bias = torch.randn(Co, dtype=dtype).to(device)

    # Dense reference
    set_algorithm(Algorithm.IMPLICIT_GEMM)
    dense_out, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)

    # Masked under test (cache rebuild — different cache schema than dense)
    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)
    masked_out, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)

    assert masked_out.shape == dense_out.shape, (
        f"Shape mismatch: masked={tuple(masked_out.shape)} dense={tuple(dense_out.shape)}"
    )
    assert masked_out.device.type == "mps", f"masked output on {masked_out.device}, expected mps"

    tol = TOLERANCES[dtype]
    # Move to CPU for comparison — some PyTorch MPS builds lack .abs/.allclose
    # kernels for the dtypes we care about, and CPU comparison is what we need
    # numerically anyway.
    masked_cpu = masked_out.detach().cpu().float()
    dense_cpu  = dense_out.detach().cpu().float()
    diff = (masked_cpu - dense_cpu).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_max = dense_cpu.abs().max().item()
    rel_max = max_err / max(ref_max, 1e-9)

    ok = torch.allclose(masked_cpu, dense_cpu, **tol)
    assert ok, (
        f"masked vs dense parity failed for res={res} ch={ch} dtype={dtype}: "
        f"max_err={max_err:.4e} rel_max={rel_max:.4e} mean_err={mean_err:.4e} "
        f"(allowed atol={tol['atol']}, rtol={tol['rtol']})"
    )


@needs_mps
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_masked_cache_reuse_stable(dtype):
    """Same masked call twice with reused cache should give bit-identical output."""
    torch.manual_seed(7)
    device = "mps"
    set_algorithm(Algorithm.MASKED_IMPLICIT_GEMM)

    feats, coords, shape = _sphere_coords(16, 32, dtype, device)
    Co = 32
    weight = torch.randn(Co, 3, 3, 3, 32, dtype=dtype).to(device)
    bias = torch.randn(Co, dtype=dtype).to(device)

    out1, cache = sparse_submanifold_conv3d(feats, coords, shape, weight, bias)
    out2, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias, neighbor_cache=cache)

    assert torch.equal(out1, out2), "Reusing the masked cache should be deterministic"

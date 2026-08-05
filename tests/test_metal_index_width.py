"""Guard against 32-bit index overflow in the Metal spconv GEMM kernels.

The kernels address every global buffer with 32-bit `uint` math
(`input[nb_idx * Ci + ci_idx]`, `output[n * Co + co]`, ...), so a buffer of 2^32
or more elements wraps the index and writes to the wrong address — silently.
Upstream FlexGEMM hit the signed-int32 form of this in its Triton kernels and
fixed it by promoting the row offset to int64 (c0a6ab0); the Metal port keeps
32-bit math (twice the headroom, no split-K term, ~67x clear of what trellis2
inference actually reaches) and enforces the bound at dispatch instead.

These tests pin that contract. The limit is monkeypatched down rather than
allocating a real 2^32-element tensor — that would be a 17 GB buffer, and the
arithmetic being checked is in the guard, not in the allocation.
"""

import platform

import pytest
import torch

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal backend is Darwin-only"
)

metal = pytest.importorskip("flex_gemm.kernels.metal")


def test_limit_is_the_unsigned_32bit_ceiling():
    # Not 2**31: the Metal kernels index with `uint`, not `int`. If this ever
    # gets "corrected" down to the Triton value, the guard starts rejecting
    # shapes the kernels handle correctly.
    assert metal._UINT32_INDEX_LIMIT == 2 ** 32


def test_guard_passes_for_realistic_trellis2_shapes():
    # Decoder worst case: the 1024-channel stage runs on a 32^3 grid and the
    # 1e6-voxel stage runs at 64 channels, so the product stays near 6e7.
    for numel in (32_768 * 1024, 1_000_000 * 64):
        assert numel < metal._UINT32_INDEX_LIMIT
    metal._assert_indexable(input=torch.empty(1024, 64), weight=torch.empty(64, 27, 64))


def test_guard_raises_before_the_index_wraps(monkeypatch):
    monkeypatch.setattr(metal, "_UINT32_INDEX_LIMIT", 1024)
    with pytest.raises(RuntimeError, match=r"32-bit index arithmetic"):
        metal._assert_indexable(input=torch.empty(64, 32))


def test_guard_names_the_offending_tensor(monkeypatch):
    monkeypatch.setattr(metal, "_UINT32_INDEX_LIMIT", 1024)
    with pytest.raises(RuntimeError, match=r"'weight'"):
        metal._assert_indexable(input=torch.empty(4, 4), weight=torch.empty(64, 32))


def test_guard_is_exclusive_at_the_limit(monkeypatch):
    # numel == limit already overflows (indices run 0..numel-1, and the wrap
    # happens at exactly `limit`), so the boundary case must be rejected.
    monkeypatch.setattr(metal, "_UINT32_INDEX_LIMIT", 1024)
    with pytest.raises(RuntimeError):
        metal._assert_indexable(input=torch.empty(1024))
    metal._assert_indexable(input=torch.empty(1023))


def test_guard_tolerates_none():
    # bias is optional and arrives as None from the conv wrappers.
    metal._assert_indexable(input=torch.empty(8), bias=None)


@pytest.mark.parametrize(
    "fn_name",
    [
        "sparse_submanifold_conv_fwd_implicit_gemm",
        "sparse_submanifold_conv_bwd_implicit_gemm",
        "sparse_submanifold_conv_fwd_masked_implicit_gemm",
        "sparse_submanifold_conv_bwd_masked_implicit_gemm",
    ],
)
def test_every_gemm_entry_point_is_guarded(fn_name, monkeypatch):
    """The guard is only worth having if no dispatch path skips it."""
    monkeypatch.setattr(metal, "_UINT32_INDEX_LIMIT", 8)
    fn = getattr(metal, fn_name)
    big = torch.empty(64)
    # Called with deliberately wrong-shaped args: the guard must fire before
    # anything reaches the extension, so we never get far enough to care.
    with pytest.raises(RuntimeError, match=r"32-bit index arithmetic"):
        if "bwd" in fn_name and "masked" in fn_name:
            fn(big, big, big, None, big, big, lambda _: big, lambda _: big, big, big, big)
        elif "bwd" in fn_name:
            fn(big, big, big, None, big)
        elif "masked" in fn_name:
            fn(big, big, None, big, big, lambda _: big, lambda _: big)
        else:
            fn(big, big, None, big)

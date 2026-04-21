"""
Metal backend for flex_gemm.

Exports 22 functions:
  - 12 C functions from _C (PyBind11 extension)
  - 10 Python-callable GEMM/weighted-sum functions from _C
"""

from . import _C

# ============================================================================
# Hash (5 functions) — from _C
# ============================================================================
hashmap_insert_cuda = _C.hashmap_insert_cuda
hashmap_lookup_cuda = _C.hashmap_lookup_cuda
hashmap_insert_3d_cuda = _C.hashmap_insert_3d_cuda
hashmap_lookup_3d_cuda = _C.hashmap_lookup_3d_cuda
hashmap_insert_3d_idx_as_val_cuda = _C.hashmap_insert_3d_idx_as_val_cuda

# ============================================================================
# Serialize (4 functions) — from _C
# ============================================================================
z_order_encode = _C.z_order_encode
z_order_decode = _C.z_order_decode
hilbert_encode = _C.hilbert_encode
hilbert_decode = _C.hilbert_decode

# ============================================================================
# Grid sample (2 functions) — from _C
# ============================================================================
hashmap_build_grid_sample_3d_nearest_neighbor_map = _C.hashmap_build_grid_sample_3d_nearest_neighbor_map
hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight = _C.hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight

# ============================================================================
# Spconv neighbor map (3 functions) — from _C
# ============================================================================
hashmap_build_submanifold_conv_neighbour_map_cuda = _C.hashmap_build_submanifold_conv_neighbour_map_cuda
neighbor_map_post_process_for_masked_implicit_gemm_1 = _C.neighbor_map_post_process_for_masked_implicit_gemm_1
neighbor_map_post_process_for_masked_implicit_gemm_2 = _C.neighbor_map_post_process_for_masked_implicit_gemm_2

# ============================================================================
# Weighted sum (2 functions) — from _C
# ============================================================================
def indice_weighed_sum_fwd(input, indices, weight):
    return _C.indice_weighted_sum_fwd(
        input.contiguous(), indices.contiguous(), weight.contiguous()
    )

def indice_weighed_sum_bwd_input(grad_output, indices, weight, N):
    return _C.indice_weighted_sum_bwd_input(
        grad_output.contiguous(), indices.contiguous(), weight.contiguous(), N
    )

# ============================================================================
# Spconv GEMM (8 functions) — Phase 6: Metal compute shader implicit GEMM
# ============================================================================
import torch

def sparse_submanifold_conv_fwd_implicit_gemm(input, weight, bias, neighbor_map):
    return _C.spconv_fwd_implicit_gemm(
        input.contiguous(), weight.contiguous(),
        bias.contiguous() if bias is not None else torch.empty(0),
        neighbor_map.contiguous()
    )

def sparse_submanifold_conv_bwd_implicit_gemm(grad_output, input, weight, bias, neighbor_map):
    grad_input, grad_weight = _C.spconv_bwd_implicit_gemm(
        grad_output.contiguous(), input.contiguous(),
        weight.contiguous(), neighbor_map.contiguous()
    )
    grad_bias = grad_output.sum(dim=0) if bias is not None else None
    return grad_input, grad_weight, grad_bias

# Split-K forward/backward currently map to the plain implicit-GEMM kernel.
# This is a true aliasing, not a silent fallback — the Metal implicit-GEMM
# kernel already has sufficient parallelism at the shapes trellis2 runs.
sparse_submanifold_conv_fwd_implicit_gemm_splitk = sparse_submanifold_conv_fwd_implicit_gemm
sparse_submanifold_conv_bwd_implicit_gemm_splitk = sparse_submanifold_conv_bwd_implicit_gemm

# Masked implicit GEMM forward calls the real Metal kernel (skips invalid V
# positions per n-block via the precomputed sorted_idx + valid_kernel +
# valid_kernel_seg). Backward still aliases to the dense kernel — masked
# backward is forward-only-relevant for trellis2 inference and lands in a
# follow-up. See FOLLOWUPS.md.
import os as _os
import warnings as _warnings

# B1 block size in the masked kernel (must match GEMM_BLOCK_N in metal/config.h).
_MASKED_B1 = 64

_MASKED_BWD_WARNED = False

def _warn_masked_bwd_once():
    global _MASKED_BWD_WARNED
    if _MASKED_BWD_WARNED:
        return
    _MASKED_BWD_WARNED = True
    if _os.environ.get("FLEX_GEMM_QUIET") == "1":
        return
    _warnings.warn(
        "flex_gemm (Metal backend): masked implicit GEMM backward is not yet "
        "implemented on Metal — falling back to the dense backward. Forward is "
        "the masked kernel. Set FLEX_GEMM_QUIET=1 to silence this.",
        stacklevel=3,
    )

def sparse_submanifold_conv_fwd_masked_implicit_gemm(input, weight, bias, neighbor_map,
                                                      sorted_idx, valid_kernel_cb, valid_kernel_seg_cb):
    valid_kernel = valid_kernel_cb(_MASKED_B1)
    valid_kernel_seg = valid_kernel_seg_cb(_MASKED_B1)
    return _C.spconv_fwd_masked_implicit_gemm(
        input.contiguous(), weight.contiguous(),
        bias.contiguous() if bias is not None else torch.empty(0),
        neighbor_map.contiguous(),
        sorted_idx.contiguous(),
        valid_kernel.contiguous(),
        valid_kernel_seg.contiguous(),
    )

def sparse_submanifold_conv_bwd_masked_implicit_gemm(grad_output, input, weight, bias, neighbor_map,
                                                      sorted_idx, valid_kernel_cb, valid_kernel_seg_cb,
                                                      valid_signal_i, valid_signal_o, valid_signal_seg):
    _warn_masked_bwd_once()
    return sparse_submanifold_conv_bwd_implicit_gemm(grad_output, input, weight, bias, neighbor_map)

sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk = sparse_submanifold_conv_fwd_masked_implicit_gemm
sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk = sparse_submanifold_conv_bwd_masked_implicit_gemm

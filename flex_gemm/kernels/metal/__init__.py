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

# All variants alias to the same kernels
sparse_submanifold_conv_fwd_implicit_gemm_splitk = sparse_submanifold_conv_fwd_implicit_gemm
sparse_submanifold_conv_bwd_implicit_gemm_splitk = sparse_submanifold_conv_bwd_implicit_gemm

def sparse_submanifold_conv_fwd_masked_implicit_gemm(input, weight, bias, neighbor_map,
                                                      sorted_idx, valid_kernel_cb, valid_kernel_seg_cb):
    return sparse_submanifold_conv_fwd_implicit_gemm(input, weight, bias, neighbor_map)

def sparse_submanifold_conv_bwd_masked_implicit_gemm(grad_output, input, weight, bias, neighbor_map,
                                                      sorted_idx, valid_kernel_cb, valid_kernel_seg_cb,
                                                      valid_signal_i, valid_signal_o, valid_signal_seg):
    return sparse_submanifold_conv_bwd_implicit_gemm(grad_output, input, weight, bias, neighbor_map)

sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk = sparse_submanifold_conv_fwd_masked_implicit_gemm
sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk = sparse_submanifold_conv_bwd_masked_implicit_gemm

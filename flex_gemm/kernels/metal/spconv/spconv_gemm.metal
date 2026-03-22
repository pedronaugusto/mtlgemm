#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "config.h"

// ============================================================================
// Implicit GEMM for sparse submanifold convolution
//
// Uses simdgroup_matrix_multiply_accumulate for hardware-accelerated 8x8 matmul.
// Three kernels: forward, backward-input, backward-weight.
//
// Block sizes: B1=64 (N-tile), B2=64 (channel-tile), BK=32 (K-tile)
// 256 threads = 8 SIMD groups, each owns one row of 8x8 sub-tiles
// Threadgroup memory: B1*BK + BK*B2 + B1 = 64*32+32*64+64 = 4160 floats = 16640 bytes
// ============================================================================

#define B1 GEMM_BLOCK_N   // 64
#define B2 GEMM_BLOCK_CO  // 64
#define BK GEMM_BLOCK_K   // 32
#define SENTINEL 0xFFFFFFFFu

// Number of 8x8 sub-tiles per dimension
#define TILES_M (B1 / 8)   // 8
#define TILES_N (B2 / 8)   // 8

// ============================================================================
// Forward: output[N, Co] = sum_v input[neighbor[n,v], :] * weight[co, v, :]
//
// Grid: (cdiv(N, B1), cdiv(Co, B2))
// Threadgroup: 256 threads (8 SIMD groups)
// ============================================================================

kernel void spconv_fwd_implicit_gemm(
    const device float* input         [[buffer(0)]],   // [N, Ci]
    const device float* weight        [[buffer(1)]],   // [Co, V, Ci]
    const device float* bias          [[buffer(2)]],   // [Co] or empty
    const device uint*  neighbor      [[buffer(3)]],   // [N, V]
    device float*       output        [[buffer(4)]],   // [N, Co]
    constant uint& N                  [[buffer(5)]],
    constant uint& Co                 [[buffer(6)]],
    constant uint& Ci                 [[buffer(7)]],
    constant uint& V                  [[buffer(8)]],
    constant uint& has_bias           [[buffer(9)]],
    threadgroup float* smem           [[threadgroup(0)]],
    uint2 gid                         [[threadgroup_position_in_grid]],
    uint  lid                         [[thread_index_in_threadgroup]],
    uint  simd_id                     [[simdgroup_index_in_threadgroup]],
    uint  lane_id                     [[thread_index_in_simdgroup]]
) {
    uint n_base  = gid.x * B1;
    uint co_base = gid.y * B2;

    uint threads = GEMM_THREADS;

    // Shared memory layout: smem_a[B1][BK] | smem_b[BK][B2] | smem_nb[B1]
    threadgroup float* smem_a  = smem;
    threadgroup float* smem_b  = smem + B1 * BK;
    threadgroup uint*  smem_nb = (threadgroup uint*)(smem + B1 * BK + BK * B2);

    // Each SIMD group owns one row of 8x8 sub-tiles (TILES_N accumulators)
    // simd_id selects which row (0..TILES_M-1)
    simdgroup_matrix<float, 8, 8> acc[TILES_N];
    for (uint i = 0; i < TILES_N; i++)
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    // Loop over volume elements
    for (uint v = 0; v < V; v++) {
        // Load B1 neighbor indices into shared memory
        for (uint i = lid; i < B1; i += threads) {
            uint n = n_base + i;
            smem_nb[i] = (n < N) ? neighbor[n * V + v] : SENTINEL;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Loop over Ci in BK-sized blocks
        for (uint bk = 0; bk < Ci; bk += BK) {
            // Cooperative load: smem_a[B1][BK] = input[neighbor[n, v], bk..bk+BK]
            for (uint i = lid; i < B1 * BK; i += threads) {
                uint row = i / BK;
                uint col = i % BK;
                uint ci_idx = bk + col;
                uint nb_idx = smem_nb[row];
                smem_a[row * BK + col] = (nb_idx != SENTINEL && ci_idx < Ci) ?
                    input[nb_idx * Ci + ci_idx] : 0.0f;
            }

            // Cooperative load: smem_b[BK][B2] = weight[co, v, bk..bk+BK]
            // weight layout: [Co, V, Ci] — weight[co, v, ci] = weight[(co * V + v) * Ci + ci]
            for (uint i = lid; i < BK * B2; i += threads) {
                uint row = i / B2;
                uint col = i % B2;
                uint ci_idx = bk + row;
                uint co_idx = co_base + col;
                smem_b[row * B2 + col] = (ci_idx < Ci && co_idx < Co) ?
                    weight[(co_idx * V + v) * Ci + ci_idx] : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // simdgroup_matrix multiply: each SIMD group does its row of 8x8 tiles
            simdgroup_matrix<float, 8, 8> a_mat;
            for (uint k8 = 0; k8 < BK; k8 += 8) {
                // Load A sub-tile: rows [simd_id*8..simd_id*8+7], cols [k8..k8+7]
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);
                for (uint tn = 0; tn < TILES_N; tn++) {
                    simdgroup_matrix<float, 8, 8> b_mat;
                    simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store results: write accumulators to shared memory, then global with bounds + bias
    for (uint tn = 0; tn < TILES_N; tn++) {
        simdgroup_store(acc[tn], smem_a + simd_id * 8 * B2 + tn * 8, B2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write from shared memory to global with bounds checking + bias
    for (uint i = lid; i < B1 * B2; i += threads) {
        uint row = i / B2;
        uint col = i % B2;
        uint n = n_base + row;
        uint co = co_base + col;
        if (n < N && co < Co) {
            float val = smem_a[row * B2 + col];
            if (has_bias) val += bias[co];
            output[n * Co + co] = val;
        }
    }
}

// ============================================================================
// Backward-input: grad_input[N, Ci] = sum_v grad_output[neighbor_inv[n,v], :] * weight^T
//
// neighbor_inv[n, v] = neighbor[n, V-1-v]  (flipped)
//
// Grid: (cdiv(N, B1), cdiv(Ci, B2))
// Threadgroup: 256 threads (8 SIMD groups)
// ============================================================================

kernel void spconv_bwd_input_implicit_gemm(
    const device float* grad_output   [[buffer(0)]],   // [N, Co]
    const device float* weight        [[buffer(1)]],   // [Co, V, Ci]
    const device uint*  neighbor      [[buffer(2)]],   // [N, V]
    device float*       grad_input    [[buffer(3)]],   // [N, Ci]
    constant uint& N                  [[buffer(4)]],
    constant uint& Co                 [[buffer(5)]],
    constant uint& Ci                 [[buffer(6)]],
    constant uint& V                  [[buffer(7)]],
    threadgroup float* smem           [[threadgroup(0)]],
    uint2 gid                         [[threadgroup_position_in_grid]],
    uint  lid                         [[thread_index_in_threadgroup]],
    uint  simd_id                     [[simdgroup_index_in_threadgroup]],
    uint  lane_id                     [[thread_index_in_simdgroup]]
) {
    uint n_base  = gid.x * B1;
    uint ci_base = gid.y * B2;

    uint threads = GEMM_THREADS;

    threadgroup float* smem_a  = smem;
    threadgroup float* smem_b  = smem + B1 * BK;
    threadgroup uint*  smem_nb = (threadgroup uint*)(smem + B1 * BK + BK * B2);

    simdgroup_matrix<float, 8, 8> acc[TILES_N];
    for (uint i = 0; i < TILES_N; i++)
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint v = 0; v < V; v++) {
        // Flipped neighbor access: neighbor[n, V-1-v]
        uint v_flip = V - 1 - v;

        for (uint i = lid; i < B1; i += threads) {
            uint n = n_base + i;
            smem_nb[i] = (n < N) ? neighbor[n * V + v_flip] : SENTINEL;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint bk = 0; bk < Co; bk += BK) {
            // smem_a[B1][BK] = grad_output[neighbor_inv[n, v], bk..bk+BK]
            for (uint i = lid; i < B1 * BK; i += threads) {
                uint row = i / BK;
                uint col = i % BK;
                uint co_idx = bk + col;
                uint nb_idx = smem_nb[row];
                smem_a[row * BK + col] = (nb_idx != SENTINEL && co_idx < Co) ?
                    grad_output[nb_idx * Co + co_idx] : 0.0f;
            }

            // smem_b[BK][B2] = weight^T for volume v
            // We want smem_b[k][ci] = weight[bk+k, v, ci_base+ci]
            for (uint i = lid; i < BK * B2; i += threads) {
                uint row = i / B2;
                uint col = i % B2;
                uint co_idx = bk + row;
                uint ci_idx = ci_base + col;
                smem_b[row * B2 + col] = (co_idx < Co && ci_idx < Ci) ?
                    weight[(co_idx * V + v) * Ci + ci_idx] : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // simdgroup_matrix multiply
            simdgroup_matrix<float, 8, 8> a_mat;
            for (uint k8 = 0; k8 < BK; k8 += 8) {
                simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);
                for (uint tn = 0; tn < TILES_N; tn++) {
                    simdgroup_matrix<float, 8, 8> b_mat;
                    simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);
                    simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store results to shared memory then global
    for (uint tn = 0; tn < TILES_N; tn++) {
        simdgroup_store(acc[tn], smem_a + simd_id * 8 * B2 + tn * 8, B2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < B1 * B2; i += threads) {
        uint row = i / B2;
        uint col = i % B2;
        uint n = n_base + row;
        uint ci = ci_base + col;
        if (n < N && ci < Ci) {
            grad_input[n * Ci + ci] = smem_a[row * B2 + col];
        }
    }
}

// ============================================================================
// Backward-weight: grad_weight[Co, V, Ci] = sum_n grad_output[n, co] * input[neighbor[n, v], ci]
//
// Grid: (cdiv(Co, B1), cdiv(V * Ci, B2))
// K dimension = N (iterate over all voxels)
// Threadgroup: 256 threads (8 SIMD groups)
// ============================================================================

kernel void spconv_bwd_weight_implicit_gemm(
    const device float* grad_output   [[buffer(0)]],   // [N, Co]
    const device float* input         [[buffer(1)]],   // [N, Ci]
    const device uint*  neighbor      [[buffer(2)]],   // [N, V]
    device float*       grad_weight   [[buffer(3)]],   // [Co, V, Ci]
    constant uint& N                  [[buffer(4)]],
    constant uint& Co                 [[buffer(5)]],
    constant uint& Ci                 [[buffer(6)]],
    constant uint& V                  [[buffer(7)]],
    threadgroup float* smem           [[threadgroup(0)]],
    uint2 gid                         [[threadgroup_position_in_grid]],
    uint  lid                         [[thread_index_in_threadgroup]],
    uint  simd_id                     [[simdgroup_index_in_threadgroup]],
    uint  lane_id                     [[thread_index_in_simdgroup]]
) {
    uint co_base = gid.x * B1;
    uint vci_base = gid.y * B2;  // flat index into V*Ci

    uint threads = GEMM_THREADS;
    uint VCi = V * Ci;

    // Shared memory: smem_a[B1][BK] for grad_output, smem_b[BK][B2] for input via neighbor
    threadgroup float* smem_a = smem;
    threadgroup float* smem_b = smem + B1 * BK;

    simdgroup_matrix<float, 8, 8> acc[TILES_N];
    for (uint i = 0; i < TILES_N; i++)
        acc[i] = simdgroup_matrix<float, 8, 8>(0.0f);

    // K dimension = N, iterate in BK-sized blocks
    for (uint bn = 0; bn < N; bn += BK) {
        // smem_a[B1][BK] = grad_output^T[co, n] = grad_output[bn+k, co_base+row]
        for (uint i = lid; i < B1 * BK; i += threads) {
            uint row = i / BK;  // co offset
            uint col = i % BK;  // n offset
            uint co_idx = co_base + row;
            uint n_idx = bn + col;
            smem_a[row * BK + col] = (co_idx < Co && n_idx < N) ?
                grad_output[n_idx * Co + co_idx] : 0.0f;
        }

        // smem_b[BK][B2] = input gathered via neighbor
        // vci_base + col maps to (v, ci): v = idx / Ci, ci = idx % Ci
        for (uint i = lid; i < BK * B2; i += threads) {
            uint row = i / B2;  // n offset
            uint col = i % B2;  // vci offset
            uint n_idx = bn + row;
            uint vci_idx = vci_base + col;
            if (n_idx < N && vci_idx < VCi) {
                uint v_idx = vci_idx / Ci;
                uint ci_idx = vci_idx % Ci;
                uint nb = neighbor[n_idx * V + v_idx];
                smem_b[row * B2 + col] = (nb != SENTINEL) ?
                    input[nb * Ci + ci_idx] : 0.0f;
            } else {
                smem_b[row * B2 + col] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // simdgroup_matrix multiply
        simdgroup_matrix<float, 8, 8> a_mat;
        for (uint k8 = 0; k8 < BK; k8 += 8) {
            simdgroup_load(a_mat, smem_a + simd_id * 8 * BK + k8, BK);
            for (uint tn = 0; tn < TILES_N; tn++) {
                simdgroup_matrix<float, 8, 8> b_mat;
                simdgroup_load(b_mat, smem_b + k8 * B2 + tn * 8, B2);
                simdgroup_multiply_accumulate(acc[tn], a_mat, b_mat, acc[tn]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results to shared memory then global
    for (uint tn = 0; tn < TILES_N; tn++) {
        simdgroup_store(acc[tn], smem_a + simd_id * 8 * B2 + tn * 8, B2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < B1 * B2; i += threads) {
        uint row = i / B2;
        uint col = i % B2;
        uint co = co_base + row;
        uint vci = vci_base + col;
        if (co < Co && vci < VCi) {
            uint v_idx = vci / Ci;
            uint ci_idx = vci % Ci;
            grad_weight[(co * V + v_idx) * Ci + ci_idx] = smem_a[row * B2 + col];
        }
    }
}

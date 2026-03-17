#include <metal_stdlib>
using namespace metal;

#include "config.h"

// ============================================================================
// Implicit GEMM for sparse submanifold convolution
//
// Fuses neighbor gather with GEMM — never materializes im2col buffer.
// Three kernels: forward, backward-input, backward-weight.
//
// Block sizes: B1=32 (N-tile), B2=32 (channel-tile), BK=32 (K-tile)
// Per-thread sub-tile: TM=4, TN=4
// Threads per threadgroup: (32/4) * (32/4) = 64
// Threadgroup memory: B1*BK + BK*B2 + B1 = 2080 floats = 8320 bytes
// ============================================================================

#define B1 GEMM_BLOCK_N   // 32
#define B2 GEMM_BLOCK_CO  // 32
#define BK GEMM_BLOCK_K   // 32
#define TM 4
#define TN 4
#define SENTINEL 0xFFFFFFFFu

// ============================================================================
// Forward: output[N, Co] = sum_v input[neighbor[n,v], :] * weight[co, v, :]
//
// Grid: (cdiv(N, B1), cdiv(Co, B2))
// Threadgroup: 64 threads (8x8)
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
    uint  lid                         [[thread_index_in_threadgroup]]
) {
    // Tile offsets
    uint n_base  = gid.x * B1;
    uint co_base = gid.y * B2;

    // Thread position within tile
    uint tx = lid % (B2 / TN);  // 0..7 (column thread)
    uint ty = lid / (B2 / TN);  // 0..7 (row thread)

    // Shared memory layout: smem_a[B1][BK] | smem_b[BK][B2] | smem_nb[B1]
    threadgroup float* smem_a  = smem;                     // B1 * BK floats
    threadgroup float* smem_b  = smem + B1 * BK;           // BK * B2 floats
    threadgroup uint*  smem_nb = (threadgroup uint*)(smem + B1 * BK + BK * B2); // B1 uints

    // Accumulator registers
    float acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    uint threads = GEMM_THREADS;  // 64

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
                if (nb_idx != SENTINEL && ci_idx < Ci) {
                    smem_a[row * BK + col] = input[nb_idx * Ci + ci_idx];
                } else {
                    smem_a[row * BK + col] = 0.0f;
                }
            }

            // Cooperative load: smem_b[BK][B2] = weight[co, v, bk..bk+BK]
            // weight layout: [Co, V, Ci] — weight[co, v, ci] = weight[(co * V + v) * Ci + ci]
            for (uint i = lid; i < BK * B2; i += threads) {
                uint row = i / B2;  // k index
                uint col = i % B2;  // co index
                uint ci_idx = bk + row;
                uint co_idx = co_base + col;
                if (ci_idx < Ci && co_idx < Co) {
                    smem_b[row * B2 + col] = weight[(co_idx * V + v) * Ci + ci_idx];
                } else {
                    smem_b[row * B2 + col] = 0.0f;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // FMA: acc[tm][tn] += smem_a[row, k] * smem_b[k, col]
            for (uint k = 0; k < BK; k++) {
                for (uint tm = 0; tm < TM; tm++) {
                    float a_val = smem_a[(ty * TM + tm) * BK + k];
                    for (uint tn = 0; tn < TN; tn++) {
                        acc[tm][tn] += a_val * smem_b[k * B2 + tx * TN + tn];
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store results + bias
    for (uint tm = 0; tm < TM; tm++) {
        uint n = n_base + ty * TM + tm;
        if (n >= N) continue;
        for (uint tn = 0; tn < TN; tn++) {
            uint co = co_base + tx * TN + tn;
            if (co >= Co) continue;
            float val = acc[tm][tn];
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
// Threadgroup: 64 threads (8x8)
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
    uint  lid                         [[thread_index_in_threadgroup]]
) {
    uint n_base  = gid.x * B1;
    uint ci_base = gid.y * B2;

    uint tx = lid % (B2 / TN);
    uint ty = lid / (B2 / TN);

    threadgroup float* smem_a  = smem;
    threadgroup float* smem_b  = smem + B1 * BK;
    threadgroup uint*  smem_nb = (threadgroup uint*)(smem + B1 * BK + BK * B2);

    float acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    uint threads = GEMM_THREADS;

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
                if (nb_idx != SENTINEL && co_idx < Co) {
                    smem_a[row * BK + col] = grad_output[nb_idx * Co + co_idx];
                } else {
                    smem_a[row * BK + col] = 0.0f;
                }
            }

            // smem_b[BK][B2] = weight^T for volume v
            // weight[co, v, ci] transposed: weight_T[co, ci] = weight[co, v, ci]
            // We want smem_b[k][ci] = weight[bk+k, v, ci_base+ci]
            for (uint i = lid; i < BK * B2; i += threads) {
                uint row = i / B2;  // k (Co dimension)
                uint col = i % B2;  // ci dimension
                uint co_idx = bk + row;
                uint ci_idx = ci_base + col;
                if (co_idx < Co && ci_idx < Ci) {
                    smem_b[row * B2 + col] = weight[(co_idx * V + v) * Ci + ci_idx];
                } else {
                    smem_b[row * B2 + col] = 0.0f;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint k = 0; k < BK; k++) {
                for (uint tm = 0; tm < TM; tm++) {
                    float a_val = smem_a[(ty * TM + tm) * BK + k];
                    for (uint tn = 0; tn < TN; tn++) {
                        acc[tm][tn] += a_val * smem_b[k * B2 + tx * TN + tn];
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store grad_input
    for (uint tm = 0; tm < TM; tm++) {
        uint n = n_base + ty * TM + tm;
        if (n >= N) continue;
        for (uint tn = 0; tn < TN; tn++) {
            uint ci = ci_base + tx * TN + tn;
            if (ci >= Ci) continue;
            grad_input[n * Ci + ci] = acc[tm][tn];
        }
    }
}

// ============================================================================
// Backward-weight: grad_weight[Co, V, Ci] = sum_n grad_output[n, co] * input[neighbor[n, v], ci]
//
// Grid: (cdiv(Co, B1), cdiv(V * Ci, B2))
// K dimension = N (iterate over all voxels)
// Threadgroup: 64 threads (8x8)
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
    uint  lid                         [[thread_index_in_threadgroup]]
) {
    uint co_base = gid.x * B1;
    uint vci_base = gid.y * B2;  // flat index into V*Ci

    uint tx = lid % (B2 / TN);
    uint ty = lid / (B2 / TN);

    // Shared memory: smem_a[B1][BK] for grad_output, smem_b[BK][B2] for input via neighbor
    threadgroup float* smem_a = smem;
    threadgroup float* smem_b = smem + B1 * BK;

    float acc[TM][TN];
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    uint threads = GEMM_THREADS;
    uint VCi = V * Ci;

    // K dimension = N, iterate in BK-sized blocks
    for (uint bn = 0; bn < N; bn += BK) {
        // smem_a[B1][BK] = grad_output^T[co, n] = grad_output[bn+k, co_base+row]
        for (uint i = lid; i < B1 * BK; i += threads) {
            uint row = i / BK;  // co offset
            uint col = i % BK;  // n offset
            uint co_idx = co_base + row;
            uint n_idx = bn + col;
            if (co_idx < Co && n_idx < N) {
                smem_a[row * BK + col] = grad_output[n_idx * Co + co_idx];
            } else {
                smem_a[row * BK + col] = 0.0f;
            }
        }

        // smem_b[BK][B2] = input gathered via neighbor
        // vci_base + col maps to (v, ci): v = idx / Ci, ci = idx % Ci
        // For each n (bn+row): gather input[neighbor[n, v], ci]
        for (uint i = lid; i < BK * B2; i += threads) {
            uint row = i / B2;  // n offset
            uint col = i % B2;  // vci offset
            uint n_idx = bn + row;
            uint vci_idx = vci_base + col;
            if (n_idx < N && vci_idx < VCi) {
                uint v_idx = vci_idx / Ci;
                uint ci_idx = vci_idx % Ci;
                uint nb = neighbor[n_idx * V + v_idx];
                if (nb != SENTINEL) {
                    smem_b[row * B2 + col] = input[nb * Ci + ci_idx];
                } else {
                    smem_b[row * B2 + col] = 0.0f;
                }
            } else {
                smem_b[row * B2 + col] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BK; k++) {
            for (uint tm = 0; tm < TM; tm++) {
                float a_val = smem_a[(ty * TM + tm) * BK + k];
                for (uint tn = 0; tn < TN; tn++) {
                    acc[tm][tn] += a_val * smem_b[k * B2 + tx * TN + tn];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store grad_weight[co, v, ci]
    for (uint tm = 0; tm < TM; tm++) {
        uint co = co_base + ty * TM + tm;
        if (co >= Co) continue;
        for (uint tn = 0; tn < TN; tn++) {
            uint vci = vci_base + tx * TN + tn;
            if (vci >= VCi) continue;
            uint v_idx = vci / Ci;
            uint ci_idx = vci % Ci;
            grad_weight[(co * V + v_idx) * Ci + ci_idx] = acc[tm][tn];
        }
    }
}

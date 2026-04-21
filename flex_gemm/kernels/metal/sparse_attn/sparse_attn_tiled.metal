#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Tiled flash-attention-v2 forward for variable-length sparse sequences.
//
// Threadgroup cooperates on a Q-tile of TILES_Q rows and streams KV tiles of
// TILES_KV rows through threadgroup shared memory. Each thread owns one Q row;
// it loads Q into registers once, then streams K/V from smem in cooperative
// loads, updating online-softmax state (m, l, O) in registers across tiles.
//
// This replaces the naive per-thread-serial-KV kernel in sparse_attn.metal
// for the case where max_seqlen is large — the naive kernel does
// O(max_kv_seqlen * H * T_q) global reads; tiled does O(max_kv_seqlen / TILES_KV
// * H * T_q) threadgroup loads plus TILES_KV-wide cooperative reads per tile,
// which is the flash-attention bandwidth win.
//
// Grid: (ceil(max_q_seqlen / TILES_Q), H, N). One threadgroup per (q-tile,
// head, sequence). TILES_Q threads per group — one simdgroup on Apple
// Silicon, gives implicit simdgroup sync for reductions.
// ============================================================================

#define TILES_Q 32
#define TILES_KV 32
#define MAX_HEAD_DIM 128
#define SENTINEL_NEG_INF -1.0e30f

#define SPARSE_ATTN_TILED_FWD_KERNEL(NAME, ELEM_T)                                         \
kernel void NAME(                                                                          \
    const device ELEM_T* q                [[buffer(0)]],                                   \
    const device ELEM_T* k                [[buffer(1)]],                                   \
    const device ELEM_T* v                [[buffer(2)]],                                   \
    const device int*    cu_seqlens_q     [[buffer(3)]],                                   \
    const device int*    cu_seqlens_kv    [[buffer(4)]],                                   \
    device ELEM_T*       out              [[buffer(5)]],                                   \
    constant uint&       H                [[buffer(6)]],                                   \
    constant uint&       C_q              [[buffer(7)]],                                   \
    constant uint&       C_v              [[buffer(8)]],                                   \
    constant float&      scale            [[buffer(9)]],                                   \
    threadgroup uchar*   smem_raw         [[threadgroup(0)]],                              \
    uint3 gid [[threadgroup_position_in_grid]],                                            \
    uint  lid [[thread_index_in_threadgroup]]                                              \
) {                                                                                        \
    uint q_tile_idx = gid.x;                                                               \
    uint h         = gid.y;                                                                \
    uint seq       = gid.z;                                                                \
                                                                                           \
    if (h >= H) return;                                                                    \
                                                                                           \
    uint q_start_global = (uint)cu_seqlens_q[seq];                                         \
    uint q_end_global   = (uint)cu_seqlens_q[seq + 1];                                     \
    uint q_len          = q_end_global - q_start_global;                                   \
                                                                                           \
    uint q_tile_start = q_tile_idx * TILES_Q;                                              \
    if (q_tile_start >= q_len) return;                                                     \
                                                                                           \
    uint kv_start_global = (uint)cu_seqlens_kv[seq];                                       \
    uint kv_end_global   = (uint)cu_seqlens_kv[seq + 1];                                   \
    uint kv_len          = kv_end_global - kv_start_global;                                \
                                                                                           \
    /* Shared memory layout: K_tile[TILES_KV][C_q] | V_tile[TILES_KV][C_v] */              \
    threadgroup ELEM_T* smem_k = (threadgroup ELEM_T*)smem_raw;                            \
    threadgroup ELEM_T* smem_v = smem_k + TILES_KV * C_q;                                  \
                                                                                           \
    /* Per-thread Q row load into registers. Threads past q_len early-mask. */             \
    uint q_local_idx = lid;  /* 0..TILES_Q-1 */                                            \
    uint q_row_in_seq = q_tile_start + q_local_idx;                                        \
    bool q_valid = q_row_in_seq < q_len;                                                   \
    uint q_row = q_start_global + q_row_in_seq;                                            \
                                                                                           \
    float q_local[MAX_HEAD_DIM];                                                           \
    for (uint c = 0; c < C_q; c++) {                                                       \
        q_local[c] = q_valid ? (float)q[(q_row * H + h) * C_q + c] : 0.0f;                 \
    }                                                                                      \
                                                                                           \
    /* Per-thread online softmax state. */                                                 \
    float m = SENTINEL_NEG_INF;                                                            \
    float l = 0.0f;                                                                        \
    float o_local[MAX_HEAD_DIM];                                                           \
    for (uint c = 0; c < C_v; c++) o_local[c] = 0.0f;                                      \
                                                                                           \
    /* Stream KV in tiles of TILES_KV. */                                                  \
    for (uint kv_tile_start = 0; kv_tile_start < kv_len; kv_tile_start += TILES_KV) {      \
        uint kv_tile_end = min(kv_tile_start + (uint)TILES_KV, kv_len);                    \
        uint tile_rows   = kv_tile_end - kv_tile_start;                                    \
                                                                                           \
        /* Cooperatively load K_tile [TILES_KV][C_q] and V_tile [TILES_KV][C_v]. */        \
        /* Each thread handles one row, looping over C. */                                 \
        if (lid < tile_rows) {                                                             \
            uint kv_row = kv_start_global + kv_tile_start + lid;                           \
            for (uint c = 0; c < C_q; c++) {                                               \
                smem_k[lid * C_q + c] = k[(kv_row * H + h) * C_q + c];                     \
            }                                                                              \
            for (uint c = 0; c < C_v; c++) {                                               \
                smem_v[lid * C_v + c] = v[(kv_row * H + h) * C_v + c];                     \
            }                                                                              \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
                                                                                           \
        /* Each thread computes S[q_local_idx][0..tile_rows] and online-softmax update. */ \
        if (q_valid) {                                                                     \
            /* Compute S values for this Q row against each KV row in the tile. */         \
            float s[TILES_KV];                                                             \
            float s_max = SENTINEL_NEG_INF;                                                \
            for (uint j = 0; j < tile_rows; j++) {                                         \
                float acc = 0.0f;                                                          \
                for (uint c = 0; c < C_q; c++) {                                           \
                    acc += q_local[c] * (float)smem_k[j * C_q + c];                        \
                }                                                                          \
                acc *= scale;                                                              \
                s[j] = acc;                                                                \
                if (acc > s_max) s_max = acc;                                              \
            }                                                                              \
                                                                                           \
            /* Online softmax merge. */                                                    \
            float m_new = max(m, s_max);                                                   \
            float alpha = exp(m - m_new);                                                  \
            /* Rescale o_local and l by alpha, add beta*V contributions. */                \
            for (uint c = 0; c < C_v; c++) o_local[c] *= alpha;                            \
            l *= alpha;                                                                    \
            for (uint j = 0; j < tile_rows; j++) {                                         \
                float beta = exp(s[j] - m_new);                                            \
                l += beta;                                                                 \
                for (uint c = 0; c < C_v; c++) {                                           \
                    o_local[c] += beta * (float)smem_v[j * C_v + c];                       \
                }                                                                          \
            }                                                                              \
            m = m_new;                                                                     \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    }                                                                                      \
                                                                                           \
    /* Normalize and write. */                                                             \
    if (q_valid) {                                                                         \
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;                                      \
        for (uint c = 0; c < C_v; c++) {                                                   \
            out[(q_row * H + h) * C_v + c] = (ELEM_T)(o_local[c] * inv_l);                 \
        }                                                                                  \
    }                                                                                      \
}

SPARSE_ATTN_TILED_FWD_KERNEL(sparse_attention_tiled_fwd, float)
SPARSE_ATTN_TILED_FWD_KERNEL(sparse_attention_tiled_fwd_half, half)
SPARSE_ATTN_TILED_FWD_KERNEL(sparse_attention_tiled_fwd_bfloat, bfloat)

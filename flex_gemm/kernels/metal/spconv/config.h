#pragma once

// Tile sizes for implicit GEMM (Phase 6)
// These are tuned for Apple M3 Max (40 GPU cores)
#define GEMM_BLOCK_N 32
#define GEMM_BLOCK_CO 32
#define GEMM_BLOCK_K 32

// Threadgroup config
#define GEMM_THREADS 64

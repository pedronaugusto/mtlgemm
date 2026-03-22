#import <torch/extension.h>
#import <Metal/Metal.h>
#import "common/metal_context.h"
#import "hash/api.h"
#import "serialize/api.h"
#import "grid_sample/api.h"
#import "spconv/api.h"

#include <dlfcn.h>
#include <chrono>
#include <mutex>

#define BLOCK_SIZE 256

// ============================================================================
// Buffer helpers — zero-copy MTLBuffers via newBufferWithBytesNoCopy
// Apple Silicon unified memory: CPU tensor data_ptr() is GPU-accessible.
// No memcpy needed — wraps existing memory directly.
// ============================================================================

namespace flex_gemm {
namespace metal {

static MetalContext& ctx() { return MetalContext::instance(); }

static id<MTLBuffer> alloc(size_t bytes) {
    return [ctx().device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

// Zero-copy: wraps tensor memory as MTLBuffer, no copies on unified memory.
// Returns (buffer, cpu_tensor) — caller must keep cpu_tensor alive to prevent
// the underlying memory from being freed while the GPU is using it.
struct TensorBuffer {
    id<MTLBuffer> buffer;
    torch::Tensor backing;  // prevents deallocation
};

static TensorBuffer from_tensor(const torch::Tensor& t) {
    auto tc = t.contiguous().cpu();
    size_t bytes = tc.nbytes();
    TORCH_CHECK(bytes > 0, "Cannot create Metal buffer from empty tensor");
    id<MTLBuffer> buf = [ctx().device() newBufferWithBytesNoCopy:tc.data_ptr()
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
    TORCH_CHECK(buf != nil, "Failed to create zero-copy Metal buffer");
    return {buf, tc};
}

// For in-place mutation: same zero-copy wrapping, but the tensor is modified in place.
static TensorBuffer from_tensor_inplace(torch::Tensor& t) {
    TORCH_CHECK(t.is_contiguous(), "In-place tensor must be contiguous");
    auto tc = t.cpu();
    size_t bytes = tc.nbytes();
    TORCH_CHECK(bytes > 0, "Cannot create Metal buffer from empty tensor");
    id<MTLBuffer> buf = [ctx().device() newBufferWithBytesNoCopy:tc.data_ptr()
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
    TORCH_CHECK(buf != nil, "Failed to create zero-copy Metal buffer");
    return {buf, tc};
}

// For output tensors: create tensor first, then wrap as zero-copy buffer.
// After GPU dispatch, tensor already contains the results — no copy back needed.
static TensorBuffer make_output(const std::vector<int64_t>& sizes, torch::ScalarType dtype) {
    auto t = torch::empty(sizes, torch::TensorOptions().dtype(dtype));
    size_t bytes = t.nbytes();
    TORCH_CHECK(bytes > 0, "Cannot create Metal buffer from empty tensor");
    id<MTLBuffer> buf = [ctx().device() newBufferWithBytesNoCopy:t.data_ptr()
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
    TORCH_CHECK(buf != nil, "Failed to create zero-copy Metal buffer");
    return {buf, t};
}

static TensorBuffer make_output_zeroed(const std::vector<int64_t>& sizes, torch::ScalarType dtype) {
    auto t = torch::zeros(sizes, torch::TensorOptions().dtype(dtype));
    size_t bytes = t.nbytes();
    TORCH_CHECK(bytes > 0, "Cannot create Metal buffer from empty tensor");
    id<MTLBuffer> buf = [ctx().device() newBufferWithBytesNoCopy:t.data_ptr()
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
    TORCH_CHECK(buf != nil, "Failed to create zero-copy Metal buffer");
    return {buf, t};
}

// Legacy compat: alloc + memset for sentinel-filled buffers
static TensorBuffer make_output_filled(const std::vector<int64_t>& sizes, torch::ScalarType dtype, int64_t fill_val) {
    auto t = torch::full(sizes, fill_val, torch::TensorOptions().dtype(dtype));
    size_t bytes = t.nbytes();
    id<MTLBuffer> buf = [ctx().device() newBufferWithBytesNoCopy:t.data_ptr()
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
    TORCH_CHECK(buf != nil, "Failed to create zero-copy Metal buffer");
    return {buf, t};
}

// ============================================================================
// Spconv GEMM autotune timing cache
// Records kernel execution times per shape to verify tile config is optimal
// and to support future multi-config autotuning.
// ============================================================================

static struct SpconvTimingCache {
    std::mutex mu;
    // Key: (N_bucket, Ci, Co, V) → avg time in microseconds (EMA)
    std::unordered_map<uint64_t, float> cache;

    static uint64_t key(uint32_t N, uint32_t Ci, uint32_t Co, uint32_t V) {
        // Bucket N to nearest power of 2 for cache coherence
        uint32_t nb = 1;
        while (nb < N) nb <<= 1;
        return (uint64_t(nb) << 48) | (uint64_t(Ci) << 32) | (uint64_t(Co) << 16) | V;
    }

    void record(uint32_t N, uint32_t Ci, uint32_t Co, uint32_t V, float us) {
        std::lock_guard<std::mutex> lock(mu);
        auto k = key(N, Ci, Co, V);
        auto it = cache.find(k);
        if (it == cache.end()) {
            cache[k] = us;
        } else {
            it->second = 0.9f * it->second + 0.1f * us; // EMA
        }
    }
} g_spconv_timing;

// ============================================================================
// Hash functions
// ============================================================================
namespace hash {

void hashmap_insert_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    const torch::Tensor& values
) {
    TORCH_CHECK(hashmap_keys.dtype() == torch::kUInt32, "Only uint32 keys supported for Metal hashmap insert");
    TORCH_CHECK(hashmap_values.dtype() == torch::kUInt32, "Only uint32 values supported");

    // Move to CPU for Metal access
    hashmap_keys = hashmap_keys.cpu().contiguous();
    hashmap_values = hashmap_values.cpu().contiguous();

    uint32_t N = (uint32_t)hashmap_keys.size(0);
    uint32_t M = (uint32_t)keys.size(0);

    auto buf_hk = from_tensor_inplace(hashmap_keys);
    auto buf_hv = from_tensor_inplace(hashmap_values);
    auto buf_k = from_tensor(keys);
    auto buf_v = from_tensor(values);

    ctx().dispatch("hashmap_insert_u32", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_hk.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_hv.buffer offset:0 atIndex:1];
        [enc setBuffer:buf_k.buffer offset:0 atIndex:2];
        [enc setBuffer:buf_v.buffer offset:0 atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&M length:sizeof(M) atIndex:5];
    }, M);

    if (hashmap_keys.data_ptr() != buf_hk.backing.data_ptr())
        memcpy(hashmap_keys.data_ptr(), buf_hk.backing.data_ptr(), hashmap_keys.nbytes());
    if (hashmap_values.data_ptr() != buf_hv.backing.data_ptr())
        memcpy(hashmap_values.data_ptr(), buf_hv.backing.data_ptr(), hashmap_values.nbytes());
}

torch::Tensor hashmap_lookup_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& keys
) {
    TORCH_CHECK(hashmap_keys.dtype() == torch::kUInt32, "Only uint32 keys supported");

    uint32_t N = (uint32_t)hashmap_keys.size(0);
    uint32_t M = (uint32_t)keys.size(0);

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_values);
    auto buf_k = from_tensor(keys);
    auto out = make_output({(int64_t)M}, hashmap_values.scalar_type());

    ctx().dispatch("hashmap_lookup_u32_u32", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_hk.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_hv.buffer offset:0 atIndex:1];
        [enc setBuffer:buf_k.buffer offset:0 atIndex:2];
        [enc setBuffer:out.buffer offset:0 atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&M length:sizeof(M) atIndex:5];
    }, M);

    return out.backing;
}

void hashmap_insert_3d_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    const torch::Tensor& values,
    int W, int H, int D
) {
    TORCH_CHECK(coords.dtype() == torch::kInt32, "Coords must be int32");

    hashmap_keys = hashmap_keys.cpu().contiguous();
    hashmap_values = hashmap_values.cpu().contiguous();
    uint32_t M = (uint32_t)coords.size(0);

    auto buf_coords = from_tensor(coords);
    auto buf_vals = from_tensor(values);

    if (hashmap_keys.dtype() == torch::kUInt64) {
        // uint64 keys: split into hi/lo uint32 buffers
        uint32_t N = (uint32_t)hashmap_keys.size(0);
        auto keys_u64 = hashmap_keys;
        auto keys_flat = keys_u64.view({-1}).contiguous();
        // Reinterpret as uint32 pairs: [N] u64 → [N*2] u32 (little-endian: lo, hi)
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)N * 2}, torch::kUInt32).clone();
        // On little-endian: keys_u32[2*i] = lo, keys_u32[2*i+1] = hi
        auto keys_lo = keys_u32.slice(0, 0, N * 2, 2).contiguous();  // even indices
        auto keys_hi = keys_u32.slice(0, 1, N * 2, 2).contiguous();  // odd indices

        auto buf_hi = from_tensor_inplace(keys_hi);
        auto buf_lo = from_tensor_inplace(keys_lo);
        auto buf_hv = from_tensor_inplace(hashmap_values);

        ctx().dispatch("hashmap_insert_3d_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_lo.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_coords.buffer offset:0 atIndex:3];
            [enc setBuffer:buf_vals.buffer offset:0 atIndex:4];
            [enc setBytes:&N length:sizeof(N) atIndex:5];
            [enc setBytes:&M length:sizeof(M) atIndex:6];
            [enc setBytes:&W length:sizeof(W) atIndex:7];
            [enc setBytes:&H length:sizeof(H) atIndex:8];
            [enc setBytes:&D length:sizeof(D) atIndex:9];
        }, M);

        if (hashmap_values.data_ptr() != buf_hv.backing.data_ptr())
            memcpy(hashmap_values.data_ptr(), buf_hv.backing.data_ptr(), hashmap_values.nbytes());
        // Rejoin hi/lo back into uint64 tensor — backing tensors already have GPU results
        auto rejoined = torch::empty({(int64_t)N * 2}, torch::kUInt32);
        // Interleave: lo at even, hi at odd
        for (uint32_t i = 0; i < N; i++) {
            ((uint32_t*)rejoined.data_ptr())[2*i]     = ((uint32_t*)buf_lo.backing.data_ptr())[i];
            ((uint32_t*)rejoined.data_ptr())[2*i + 1] = ((uint32_t*)buf_hi.backing.data_ptr())[i];
        }
        memcpy(hashmap_keys.data_ptr(), rejoined.data_ptr(), hashmap_keys.nbytes());
    } else {
        TORCH_CHECK(hashmap_keys.dtype() == torch::kUInt32, "Keys must be uint32 or uint64");
        uint32_t N = (uint32_t)hashmap_keys.size(0);

        auto buf_hk = from_tensor_inplace(hashmap_keys);
        auto buf_hv = from_tensor_inplace(hashmap_values);

        ctx().dispatch("hashmap_insert_3d_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_coords.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_vals.buffer offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&M length:sizeof(M) atIndex:5];
            [enc setBytes:&W length:sizeof(W) atIndex:6];
            [enc setBytes:&H length:sizeof(H) atIndex:7];
            [enc setBytes:&D length:sizeof(D) atIndex:8];
        }, M);

        if (hashmap_keys.data_ptr() != buf_hk.backing.data_ptr())
            memcpy(hashmap_keys.data_ptr(), buf_hk.backing.data_ptr(), hashmap_keys.nbytes());
        if (hashmap_values.data_ptr() != buf_hv.backing.data_ptr())
            memcpy(hashmap_values.data_ptr(), buf_hv.backing.data_ptr(), hashmap_values.nbytes());
    }
}

torch::Tensor hashmap_lookup_3d_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    int W, int H, int D
) {
    uint32_t N = (uint32_t)hashmap_keys.size(0);
    uint32_t M = (uint32_t)coords.size(0);

    auto buf_coords = from_tensor(coords);
    auto out = make_output({(int64_t)M}, hashmap_values.scalar_type());

    if (hashmap_keys.dtype() == torch::kUInt64) {
        // uint64 keys: split into hi/lo uint32 buffers for lookup
        auto keys_flat = hashmap_keys.contiguous().view({-1});
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)N * 2}, torch::kUInt32).clone();
        auto keys_lo = keys_u32.slice(0, 0, N * 2, 2).contiguous();
        auto keys_hi = keys_u32.slice(0, 1, N * 2, 2).contiguous();

        auto buf_hi = from_tensor(keys_hi);
        auto buf_lo = from_tensor(keys_lo);
        auto buf_hv = from_tensor(hashmap_values);

        ctx().dispatch("hashmap_lookup_3d_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_lo.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_coords.buffer offset:0 atIndex:3];
            [enc setBuffer:out.buffer offset:0 atIndex:4];
            [enc setBytes:&N length:sizeof(N) atIndex:5];
            [enc setBytes:&M length:sizeof(M) atIndex:6];
            [enc setBytes:&W length:sizeof(W) atIndex:7];
            [enc setBytes:&H length:sizeof(H) atIndex:8];
            [enc setBytes:&D length:sizeof(D) atIndex:9];
        }, M);
    } else {
        TORCH_CHECK(hashmap_keys.dtype() == torch::kUInt32, "Keys must be uint32 or uint64");

        auto buf_hk = from_tensor(hashmap_keys);
        auto buf_hv = from_tensor(hashmap_values);

        ctx().dispatch("hashmap_lookup_3d_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_coords.buffer offset:0 atIndex:2];
            [enc setBuffer:out.buffer offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&M length:sizeof(M) atIndex:5];
            [enc setBytes:&W length:sizeof(W) atIndex:6];
            [enc setBytes:&H length:sizeof(H) atIndex:7];
            [enc setBytes:&D length:sizeof(D) atIndex:8];
        }, M);
    }

    return out.backing;
}

void hashmap_insert_3d_idx_as_val_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    int W, int H, int D
) {
    hashmap_keys = hashmap_keys.cpu().contiguous();
    hashmap_values = hashmap_values.cpu().contiguous();
    uint32_t M = (uint32_t)coords.size(0);

    auto buf_coords = from_tensor(coords);

    if (hashmap_keys.dtype() == torch::kUInt64) {
        uint32_t N = (uint32_t)hashmap_keys.size(0);
        auto keys_flat = hashmap_keys.view({-1}).contiguous();
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)N * 2}, torch::kUInt32).clone();
        auto keys_lo = keys_u32.slice(0, 0, N * 2, 2).contiguous();
        auto keys_hi = keys_u32.slice(0, 1, N * 2, 2).contiguous();

        auto buf_hi = from_tensor_inplace(keys_hi);
        auto buf_lo = from_tensor_inplace(keys_lo);
        auto buf_hv = from_tensor_inplace(hashmap_values);

        ctx().dispatch("hashmap_insert_3d_idx_as_val_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_lo.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_coords.buffer offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&M length:sizeof(M) atIndex:5];
            [enc setBytes:&W length:sizeof(W) atIndex:6];
            [enc setBytes:&H length:sizeof(H) atIndex:7];
            [enc setBytes:&D length:sizeof(D) atIndex:8];
        }, M);

        if (hashmap_values.data_ptr() != buf_hv.backing.data_ptr())
            memcpy(hashmap_values.data_ptr(), buf_hv.backing.data_ptr(), hashmap_values.nbytes());
        // Rejoin hi/lo back into uint64 tensor — backing tensors already have GPU results
        auto rejoined = torch::empty({(int64_t)N * 2}, torch::kUInt32);
        for (uint32_t i = 0; i < N; i++) {
            ((uint32_t*)rejoined.data_ptr())[2*i]     = ((uint32_t*)buf_lo.backing.data_ptr())[i];
            ((uint32_t*)rejoined.data_ptr())[2*i + 1] = ((uint32_t*)buf_hi.backing.data_ptr())[i];
        }
        memcpy(hashmap_keys.data_ptr(), rejoined.data_ptr(), hashmap_keys.nbytes());
    } else {
        TORCH_CHECK(hashmap_keys.dtype() == torch::kUInt32, "Keys must be uint32 or uint64");
        uint32_t N = (uint32_t)hashmap_keys.size(0);

        auto buf_hk = from_tensor_inplace(hashmap_keys);
        auto buf_hv = from_tensor_inplace(hashmap_values);

        ctx().dispatch("hashmap_insert_3d_idx_as_val_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_coords.buffer offset:0 atIndex:2];
            [enc setBytes:&N length:sizeof(N) atIndex:3];
            [enc setBytes:&M length:sizeof(M) atIndex:4];
            [enc setBytes:&W length:sizeof(W) atIndex:5];
            [enc setBytes:&H length:sizeof(H) atIndex:6];
            [enc setBytes:&D length:sizeof(D) atIndex:7];
        }, M);

        if (hashmap_keys.data_ptr() != buf_hk.backing.data_ptr())
            memcpy(hashmap_keys.data_ptr(), buf_hk.backing.data_ptr(), hashmap_keys.nbytes());
        if (hashmap_values.data_ptr() != buf_hv.backing.data_ptr())
            memcpy(hashmap_values.data_ptr(), buf_hv.backing.data_ptr(), hashmap_values.nbytes());
    }
}

} // namespace hash

// ============================================================================
// Serialize functions
// ============================================================================
namespace serialize {

void z_order_encode(
    const torch::Tensor& coords,
    const size_t bit_length,
    torch::Tensor& codes
) {
    uint32_t N_val = (uint32_t)coords.size(0);
    uint32_t bl = (uint32_t)bit_length;

    auto buf_coords = from_tensor(coords);
    codes = codes.cpu().contiguous();
    auto buf_codes = from_tensor_inplace(codes);

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "z_order_encode_3d_u32" : "z_order_encode_3d_u64";

    ctx().dispatch(kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_coords.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_codes.buffer offset:0 atIndex:1];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:2];
        [enc setBytes:&bl length:sizeof(bl) atIndex:3];
    }, N_val);

    if (codes.data_ptr() != buf_codes.backing.data_ptr())
        memcpy(codes.data_ptr(), buf_codes.backing.data_ptr(), codes.nbytes());
}

torch::Tensor z_order_decode(
    const torch::Tensor& codes,
    const size_t bit_length
) {
    uint32_t N_val = (uint32_t)codes.size(0);
    uint32_t bl = (uint32_t)bit_length;

    auto buf_codes = from_tensor(codes);
    auto out = make_output({codes.size(0), 4}, torch::kInt32);

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "z_order_decode_3d_u32" : "z_order_decode_3d_u64";

    ctx().dispatch(kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_codes.buffer offset:0 atIndex:0];
        [enc setBuffer:out.buffer offset:0 atIndex:1];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:2];
        [enc setBytes:&bl length:sizeof(bl) atIndex:3];
    }, N_val);

    return out.backing;
}

void hilbert_encode(
    const torch::Tensor& coords,
    const size_t bit_length,
    torch::Tensor& codes
) {
    uint32_t N_val = (uint32_t)coords.size(0);
    uint32_t bl = (uint32_t)bit_length;

    auto buf_coords = from_tensor(coords);
    codes = codes.cpu().contiguous();
    auto buf_codes = from_tensor_inplace(codes);

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "hilbert_encode_3d_u32" : "hilbert_encode_3d_u64";

    ctx().dispatch(kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_coords.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_codes.buffer offset:0 atIndex:1];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:2];
        [enc setBytes:&bl length:sizeof(bl) atIndex:3];
    }, N_val);

    if (codes.data_ptr() != buf_codes.backing.data_ptr())
        memcpy(codes.data_ptr(), buf_codes.backing.data_ptr(), codes.nbytes());
}

torch::Tensor hilbert_decode(
    const torch::Tensor& codes,
    const size_t bit_length
) {
    uint32_t N_val = (uint32_t)codes.size(0);
    uint32_t bl = (uint32_t)bit_length;

    auto buf_codes = from_tensor(codes);
    auto out = make_output({codes.size(0), 4}, torch::kInt32);

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "hilbert_decode_3d_u32" : "hilbert_decode_3d_u64";

    ctx().dispatch(kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_codes.buffer offset:0 atIndex:0];
        [enc setBuffer:out.buffer offset:0 atIndex:1];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:2];
        [enc setBytes:&bl length:sizeof(bl) atIndex:3];
    }, N_val);

    return out.backing;
}

} // namespace serialize

// ============================================================================
// Grid sample functions
// ============================================================================
namespace grid_sample {

torch::Tensor hashmap_build_grid_sample_3d_nearest_neighbor_map(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const torch::Tensor& grid,
    int W, int H, int D
) {
    // Insert coords into hashmap first
    hash::hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D);

    uint32_t N = (uint32_t)hashmap_keys.size(0);
    uint32_t B = (uint32_t)grid.size(0);
    uint32_t L = (uint32_t)grid.size(1);

    auto buf_neigh = make_output_filled({(int64_t)B, (int64_t)L}, torch::kUInt32, (int64_t)0xFFFFFFFF);

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_vals);
    auto buf_grid = from_tensor(grid);

    if (hashmap_keys.dtype() == torch::kUInt64) {
        auto keys_flat = hashmap_keys.contiguous().view({-1});
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)N * 2}, torch::kUInt32).clone();
        auto klo = keys_u32.slice(0, 0, N * 2, 2).contiguous();
        auto khi = keys_u32.slice(0, 1, N * 2, 2).contiguous();
        auto buf_hi = from_tensor(khi);
        auto buf_lo = from_tensor(klo);

        ctx().dispatch("grid_sample_nearest_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_lo.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_grid.buffer offset:0 atIndex:3];
            [enc setBuffer:buf_neigh.buffer offset:0 atIndex:4];
            [enc setBytes:&N length:sizeof(N) atIndex:5];
            [enc setBytes:&B length:sizeof(B) atIndex:6];
            [enc setBytes:&L length:sizeof(L) atIndex:7];
            [enc setBytes:&W length:sizeof(W) atIndex:8];
            [enc setBytes:&H length:sizeof(H) atIndex:9];
            [enc setBytes:&D length:sizeof(D) atIndex:10];
        }, B * L);
    } else {
        ctx().dispatch("grid_sample_nearest_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_grid.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_neigh.buffer offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&B length:sizeof(B) atIndex:5];
            [enc setBytes:&L length:sizeof(L) atIndex:6];
            [enc setBytes:&W length:sizeof(W) atIndex:7];
            [enc setBytes:&H length:sizeof(H) atIndex:8];
            [enc setBytes:&D length:sizeof(D) atIndex:9];
        }, B * L);
    }

    return buf_neigh.backing;
}

std::tuple<torch::Tensor, torch::Tensor> hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    const torch::Tensor& grid,
    int W, int H, int D
) {
    hash::hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D);

    uint32_t N = (uint32_t)hashmap_keys.size(0);
    uint32_t B = (uint32_t)grid.size(0);
    uint32_t L = (uint32_t)grid.size(1);

    auto buf_neigh = make_output_filled({(int64_t)B, (int64_t)L, 8}, torch::kUInt32, (int64_t)0xFFFFFFFF);
    auto buf_weight = make_output_zeroed({(int64_t)B, (int64_t)L, 8}, torch::kFloat32);

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_vals);
    auto buf_grid = from_tensor(grid);

    if (hashmap_keys.dtype() == torch::kUInt64) {
        auto keys_flat = hashmap_keys.contiguous().view({-1});
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)N * 2}, torch::kUInt32).clone();
        auto klo = keys_u32.slice(0, 0, N * 2, 2).contiguous();
        auto khi = keys_u32.slice(0, 1, N * 2, 2).contiguous();
        auto buf_hi = from_tensor(khi);
        auto buf_lo = from_tensor(klo);

        ctx().dispatch("grid_sample_trilinear_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_lo.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_grid.buffer offset:0 atIndex:3];
            [enc setBuffer:buf_neigh.buffer offset:0 atIndex:4];
            [enc setBuffer:buf_weight.buffer offset:0 atIndex:5];
            [enc setBytes:&N length:sizeof(N) atIndex:6];
            [enc setBytes:&B length:sizeof(B) atIndex:7];
            [enc setBytes:&L length:sizeof(L) atIndex:8];
            [enc setBytes:&W length:sizeof(W) atIndex:9];
            [enc setBytes:&H length:sizeof(H) atIndex:10];
            [enc setBytes:&D length:sizeof(D) atIndex:11];
        }, B * L);
    } else {
        ctx().dispatch("grid_sample_trilinear_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_grid.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_neigh.buffer offset:0 atIndex:3];
            [enc setBuffer:buf_weight.buffer offset:0 atIndex:4];
            [enc setBytes:&N length:sizeof(N) atIndex:5];
            [enc setBytes:&B length:sizeof(B) atIndex:6];
            [enc setBytes:&L length:sizeof(L) atIndex:7];
            [enc setBytes:&W length:sizeof(W) atIndex:8];
            [enc setBytes:&H length:sizeof(H) atIndex:9];
            [enc setBytes:&D length:sizeof(D) atIndex:10];
        }, B * L);
    }

    return std::make_tuple(buf_neigh.backing, buf_weight.backing);
}

torch::Tensor indice_weighted_sum_fwd(
    const torch::Tensor& input,
    const torch::Tensor& indices,
    const torch::Tensor& weight
) {
    uint32_t M = (uint32_t)indices.size(0);
    uint32_t C = (uint32_t)input.size(1);
    uint32_t V = (uint32_t)weight.size(1);

    auto buf_in = from_tensor(input);
    auto buf_idx = from_tensor(indices);
    auto buf_w = from_tensor(weight);
    auto out = make_output({(int64_t)M, (int64_t)C}, input.scalar_type());

    uint32_t tg_x = std::min(C, (uint32_t)32);
    uint32_t tg_y = std::min(M, (uint32_t)(256 / tg_x));
    MTLSize grid = MTLSizeMake((C + tg_x - 1) / tg_x, (M + tg_y - 1) / tg_y, 1);
    MTLSize threadgroup = MTLSizeMake(tg_x, tg_y, 1);

    ctx().dispatch_2d("indice_weighted_sum_fwd", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_in.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_idx.buffer offset:0 atIndex:1];
        [enc setBuffer:buf_w.buffer offset:0 atIndex:2];
        [enc setBuffer:out.buffer offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(M) atIndex:4];
        [enc setBytes:&C length:sizeof(C) atIndex:5];
        [enc setBytes:&V length:sizeof(V) atIndex:6];
    }, grid, threadgroup);

    return out.backing;
}

torch::Tensor indice_weighted_sum_bwd_input(
    const torch::Tensor& grad_output,
    const torch::Tensor& indices,
    const torch::Tensor& weight,
    int64_t N
) {
    uint32_t M = (uint32_t)indices.size(0);
    uint32_t C = (uint32_t)grad_output.size(1);
    uint32_t V = (uint32_t)weight.size(1);

    auto buf_go = from_tensor(grad_output);
    auto buf_idx = from_tensor(indices);
    auto buf_w = from_tensor(weight);
    auto out = make_output_zeroed({N, (int64_t)C}, grad_output.scalar_type());

    uint32_t MV = M * V;
    uint32_t tg_x = std::min(C, (uint32_t)32);
    uint32_t tg_y = std::min(MV, (uint32_t)(256 / tg_x));
    MTLSize grid = MTLSizeMake((C + tg_x - 1) / tg_x, (MV + tg_y - 1) / tg_y, 1);
    MTLSize threadgroup = MTLSizeMake(tg_x, tg_y, 1);

    ctx().dispatch_2d("indice_weighted_sum_bwd_input", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_go.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_idx.buffer offset:0 atIndex:1];
        [enc setBuffer:buf_w.buffer offset:0 atIndex:2];
        [enc setBuffer:out.buffer offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(M) atIndex:4];
        [enc setBytes:&C length:sizeof(C) atIndex:5];
        [enc setBytes:&V length:sizeof(V) atIndex:6];
    }, grid, threadgroup);

    return out.backing;
}

} // namespace grid_sample

// ============================================================================
// Spconv functions
// ============================================================================
namespace spconv {

torch::Tensor hashmap_build_submanifold_conv_neighbour_map_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_vals,
    const torch::Tensor& coords,
    int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Dw, int Dh, int Dd
) {
    @autoreleasepool {
    int V = Kw * Kh * Kd;

    hash::hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D);

    uint32_t hash_N = (uint32_t)hashmap_keys.size(0);
    uint32_t M = (uint32_t)coords.size(0);
    uint64_t thread_count = (uint64_t)M * (V / 2 + 1);

    auto buf_neigh = make_output_filled({coords.size(0), (int64_t)V}, torch::kUInt32, (int64_t)0xFFFFFFFF);

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_vals);
    auto buf_coords = from_tensor(coords);

    if (hashmap_keys.dtype() == torch::kUInt64) {
        auto keys_flat = hashmap_keys.contiguous().view({-1});
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)hash_N * 2}, torch::kUInt32).clone();
        auto klo = keys_u32.slice(0, 0, hash_N * 2, 2).contiguous();
        auto khi = keys_u32.slice(0, 1, hash_N * 2, 2).contiguous();
        auto buf_hi = from_tensor(khi);
        auto buf_lo = from_tensor(klo);

        ctx().dispatch("submanifold_conv_neighbor_map_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_lo.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_coords.buffer offset:0 atIndex:3];
            [enc setBuffer:buf_neigh.buffer offset:0 atIndex:4];
            [enc setBytes:&hash_N length:sizeof(hash_N) atIndex:5];
            [enc setBytes:&M length:sizeof(M) atIndex:6];
            [enc setBytes:&W length:sizeof(W) atIndex:7];
            [enc setBytes:&H length:sizeof(H) atIndex:8];
            [enc setBytes:&D length:sizeof(D) atIndex:9];
            [enc setBytes:&V length:sizeof(V) atIndex:10];
            [enc setBytes:&Kw length:sizeof(Kw) atIndex:11];
            [enc setBytes:&Kh length:sizeof(Kh) atIndex:12];
            [enc setBytes:&Kd length:sizeof(Kd) atIndex:13];
            [enc setBytes:&Dw length:sizeof(Dw) atIndex:14];
            [enc setBytes:&Dh length:sizeof(Dh) atIndex:15];
            [enc setBytes:&Dd length:sizeof(Dd) atIndex:16];
        }, thread_count);
    } else {
        ctx().dispatch("submanifold_conv_neighbor_map_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_coords.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_neigh.buffer offset:0 atIndex:3];
            [enc setBytes:&hash_N length:sizeof(hash_N) atIndex:4];
            [enc setBytes:&M length:sizeof(M) atIndex:5];
            [enc setBytes:&W length:sizeof(W) atIndex:6];
            [enc setBytes:&H length:sizeof(H) atIndex:7];
            [enc setBytes:&D length:sizeof(D) atIndex:8];
            [enc setBytes:&V length:sizeof(V) atIndex:9];
            [enc setBytes:&Kw length:sizeof(Kw) atIndex:10];
            [enc setBytes:&Kh length:sizeof(Kh) atIndex:11];
            [enc setBytes:&Kd length:sizeof(Kd) atIndex:12];
            [enc setBytes:&Dw length:sizeof(Dw) atIndex:13];
            [enc setBytes:&Dh length:sizeof(Dh) atIndex:14];
            [enc setBytes:&Dd length:sizeof(Dd) atIndex:15];
        }, thread_count);
    }

    return buf_neigh.backing;
    } // @autoreleasepool
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
neighbor_map_post_process_for_masked_implicit_gemm_1(
    const torch::Tensor& neighbor_map
) {
    @autoreleasepool {
    int64_t N = neighbor_map.size(0);
    int64_t V = neighbor_map.size(1);

    uint32_t N32 = (uint32_t)N;
    uint32_t V32 = (uint32_t)V;
    uint32_t tg_size = 256;
    uint32_t num_groups = (N32 + tg_size - 1) / tg_size;
    uint32_t shared_mem = tg_size * V32 * sizeof(uint32_t);

    auto buf_nm = from_tensor(neighbor_map);
    auto buf_gc = make_output({N}, torch::kInt32);
    auto buf_bc = make_output({N}, torch::kInt32);
    auto buf_nmt = make_output({V * N}, torch::kUInt32);
    auto buf_nmaskt = make_output({V * N}, torch::kInt32);

    // Dispatch gray_code kernel — must complete before CPU argsort/cumsum
    ctx().begin_batch();
    ctx().dispatch_2d_batched("neighbor_map_gray_code", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_nm.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_gc.buffer offset:0 atIndex:1];
        [enc setBuffer:buf_bc.buffer offset:0 atIndex:2];
        [enc setBuffer:buf_nmt.buffer offset:0 atIndex:3];
        [enc setBuffer:buf_nmaskt.buffer offset:0 atIndex:4];
        [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
        [enc setBytes:&V32 length:sizeof(V32) atIndex:6];
        [enc setThreadgroupMemoryLength:shared_mem atIndex:0];
    }, MTLSizeMake(num_groups, 1, 1), MTLSizeMake(tg_size, 1, 1));
    ctx().end_batch();

    auto sorted_idx = torch::argsort(buf_bc.backing);

    // Prefix sum and gather
    auto prefix_sum = torch::cumsum(buf_nmaskt.backing, 0, torch::kInt32);
    auto total_valid = prefix_sum[-1].item<int32_t>();

    auto buf_ps = from_tensor(prefix_sum);
    auto buf_nmt2 = from_tensor(buf_nmt.backing);
    auto buf_vso = make_output({total_valid}, torch::kUInt32);
    auto buf_vsi = make_output({total_valid}, torch::kUInt32);
    auto buf_vss = make_output({V + 1}, torch::kUInt32);

    ctx().begin_batch();
    ctx().dispatch_batched("gather_idx_val_seg_from_prefix_sum", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_ps.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_nmt2.buffer offset:0 atIndex:1];
        [enc setBuffer:buf_vso.buffer offset:0 atIndex:2];
        [enc setBuffer:buf_vsi.buffer offset:0 atIndex:3];
        [enc setBuffer:buf_vss.buffer offset:0 atIndex:4];
        [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
        [enc setBytes:&V32 length:sizeof(V32) atIndex:6];
    }, N * V);
    ctx().end_batch();

    return std::make_tuple(buf_gc.backing, sorted_idx, buf_vsi.backing, buf_vso.backing, buf_vss.backing);
    } // @autoreleasepool
}

std::tuple<torch::Tensor, torch::Tensor>
neighbor_map_post_process_for_masked_implicit_gemm_2(
    const torch::Tensor& gray_code,
    const torch::Tensor& sorted_idx,
    int block_size
) {
    @autoreleasepool {
    uint32_t N = (uint32_t)gray_code.size(0);
    auto num_blocks = (int64_t)((N + block_size - 1) / block_size);

    int block_dim = block_size;
    uint32_t tg_size = 256;
    uint32_t num_dispatch_groups = ((N + 1) / 2 + tg_size - 1) / tg_size;
    uint32_t shared_mem = tg_size * sizeof(int32_t);

    auto buf_gc = from_tensor(gray_code);
    auto buf_si = from_tensor(sorted_idx);
    auto buf_rc = make_output({num_blocks}, torch::kInt32);
    auto buf_sl = make_output({num_blocks + 1}, torch::kInt32);

    // Dispatch reduce_code kernel — must complete before CPU cumsum
    ctx().begin_batch();
    ctx().dispatch_2d_batched("reduce_code", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_gc.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_si.buffer offset:0 atIndex:1];
        [enc setBuffer:buf_rc.buffer offset:0 atIndex:2];
        [enc setBuffer:buf_sl.buffer offset:0 atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&block_dim length:sizeof(block_dim) atIndex:5];
        [enc setThreadgroupMemoryLength:shared_mem atIndex:0];
    }, MTLSizeMake(num_dispatch_groups, 1, 1), MTLSizeMake(tg_size, 1, 1));
    ctx().end_batch();

    auto seglen = torch::cumsum(buf_sl.backing, 0, torch::kInt32);

    auto total_valid = seglen[-1].item<int32_t>();
    uint32_t nb = (uint32_t)num_blocks;

    auto buf_sl2 = from_tensor(seglen);
    auto buf_rc2 = from_tensor(buf_rc.backing);
    auto buf_vki = make_output({total_valid}, torch::kInt32);

    ctx().begin_batch();
    ctx().dispatch_batched("scatter_reduced_code", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_rc2.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_sl2.buffer offset:0 atIndex:1];
        [enc setBuffer:buf_vki.buffer offset:0 atIndex:2];
        [enc setBytes:&nb length:sizeof(nb) atIndex:3];
    }, num_blocks);
    ctx().end_batch();

    return std::make_tuple(buf_vki.backing, seglen);
    } // @autoreleasepool
}

// ============================================================================
// Spconv implicit GEMM — Metal compute shader dispatch
// ============================================================================

torch::Tensor spconv_fwd_implicit_gemm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& neighbor
) {
    uint32_t N  = (uint32_t)input.size(0);
    uint32_t Ci = (uint32_t)input.size(1);
    uint32_t Co = (uint32_t)weight.size(0);
    uint32_t V  = (uint32_t)weight.size(1);
    // weight shape: [Co, V, Ci]

    auto buf_input    = from_tensor(input);
    auto buf_weight   = from_tensor(weight);
    auto buf_neighbor = from_tensor(neighbor);
    auto out = make_output({(int64_t)N, (int64_t)Co}, input.scalar_type());

    // Bias: use zero-copy if present, dummy alloc otherwise
    TensorBuffer buf_bias_tb;
    id<MTLBuffer> bias_buf;
    if (bias.numel() > 0) {
        buf_bias_tb = from_tensor(bias);
        bias_buf = buf_bias_tb.buffer;
    } else {
        bias_buf = alloc(4);
    }
    uint32_t has_bias = (bias.numel() > 0) ? 1 : 0;

    // Grid: (cdiv(N, 64), cdiv(Co, 64)), Threadgroup: 256
    uint32_t grid_x = (N + 63) / 64;
    uint32_t grid_y = (Co + 63) / 64;
    // Shared memory: B1*BK + BK*B2 + B1 = 64*32 + 32*64 + 64 = 4160 floats = 16640 bytes
    uint32_t shared_mem = (64 * 32 + 32 * 64 + 64) * sizeof(float);

    auto pso = ctx().pipeline("spconv_fwd_implicit_gemm");

    @autoreleasepool {
        id<MTLCommandBuffer> cmdbuf = [ctx().queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_input.buffer offset:0 atIndex:0];
        [enc setBuffer:buf_weight.buffer offset:0 atIndex:1];
        [enc setBuffer:bias_buf offset:0 atIndex:2];
        [enc setBuffer:buf_neighbor.buffer offset:0 atIndex:3];
        [enc setBuffer:out.buffer offset:0 atIndex:4];
        [enc setBytes:&N length:sizeof(N) atIndex:5];
        [enc setBytes:&Co length:sizeof(Co) atIndex:6];
        [enc setBytes:&Ci length:sizeof(Ci) atIndex:7];
        [enc setBytes:&V length:sizeof(V) atIndex:8];
        [enc setBytes:&has_bias length:sizeof(has_bias) atIndex:9];
        [enc setThreadgroupMemoryLength:shared_mem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];

        auto t0 = std::chrono::high_resolution_clock::now();
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1 - t0).count();
        g_spconv_timing.record(N, Ci, Co, V, us);
    }

    return out.backing;
}

std::tuple<torch::Tensor, torch::Tensor> spconv_bwd_implicit_gemm(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& neighbor
) {
    uint32_t N  = (uint32_t)input.size(0);
    uint32_t Ci = (uint32_t)input.size(1);
    uint32_t Co = (uint32_t)weight.size(0);
    uint32_t V  = (uint32_t)weight.size(1);
    uint32_t VCi = V * Ci;

    auto buf_go       = from_tensor(grad_output);
    auto buf_input    = from_tensor(input);
    auto buf_weight   = from_tensor(weight);
    auto buf_neighbor = from_tensor(neighbor);
    auto out_gi       = make_output({(int64_t)N, (int64_t)Ci}, input.scalar_type());
    auto out_gw       = make_output({(int64_t)Co, (int64_t)V, (int64_t)Ci}, weight.scalar_type());

    // Shared memory for bwd_input: B1*BK + BK*B2 + B1 (smem_nb) = 4160 floats = 16640 bytes
    uint32_t shared_mem_input = (64 * 32 + 32 * 64 + 64) * sizeof(float);
    // Shared memory for bwd_weight: B1*BK + BK*B2 = 4096 floats = 16384 bytes
    uint32_t shared_mem_weight = (64 * 32 + 32 * 64) * sizeof(float);

    auto pso_gi = ctx().pipeline("spconv_bwd_input_implicit_gemm");
    auto pso_gw = ctx().pipeline("spconv_bwd_weight_implicit_gemm");

    @autoreleasepool {
        id<MTLCommandBuffer> cmdbuf = [ctx().queue() commandBuffer];

        // Backward-input kernel
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:pso_gi];
            [enc setBuffer:buf_go.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_weight.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_neighbor.buffer offset:0 atIndex:2];
            [enc setBuffer:out_gi.buffer offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&Co length:sizeof(Co) atIndex:5];
            [enc setBytes:&Ci length:sizeof(Ci) atIndex:6];
            [enc setBytes:&V length:sizeof(V) atIndex:7];
            [enc setThreadgroupMemoryLength:shared_mem_input atIndex:0];
            uint32_t grid_x = (N + 63) / 64;
            uint32_t grid_y = (Ci + 63) / 64;
            [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Backward-weight kernel
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:pso_gw];
            [enc setBuffer:buf_go.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_input.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_neighbor.buffer offset:0 atIndex:2];
            [enc setBuffer:out_gw.buffer offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&Co length:sizeof(Co) atIndex:5];
            [enc setBytes:&Ci length:sizeof(Ci) atIndex:6];
            [enc setBytes:&V length:sizeof(V) atIndex:7];
            [enc setThreadgroupMemoryLength:shared_mem_weight atIndex:0];
            uint32_t grid_x = (Co + 63) / 64;
            uint32_t grid_y = (VCi + 63) / 64;
            [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }

    return std::make_tuple(out_gi.backing, out_gw.backing);
}

} // namespace spconv

} // namespace metal
} // namespace flex_gemm

// ============================================================================
// PyBind11 module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace flex_gemm::metal;

    // Hash (5)
    m.def("hashmap_insert_cuda", &hash::hashmap_insert_cuda);
    m.def("hashmap_lookup_cuda", &hash::hashmap_lookup_cuda);
    m.def("hashmap_insert_3d_cuda", &hash::hashmap_insert_3d_cuda);
    m.def("hashmap_lookup_3d_cuda", &hash::hashmap_lookup_3d_cuda);
    m.def("hashmap_insert_3d_idx_as_val_cuda", &hash::hashmap_insert_3d_idx_as_val_cuda);

    // Serialize (4)
    m.def("z_order_encode", &serialize::z_order_encode);
    m.def("z_order_decode", &serialize::z_order_decode);
    m.def("hilbert_encode", &serialize::hilbert_encode);
    m.def("hilbert_decode", &serialize::hilbert_decode);

    // Grid sample (2 + 2 weighted sum)
    m.def("hashmap_build_grid_sample_3d_nearest_neighbor_map", &grid_sample::hashmap_build_grid_sample_3d_nearest_neighbor_map);
    m.def("hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight", &grid_sample::hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight);
    m.def("indice_weighted_sum_fwd", &grid_sample::indice_weighted_sum_fwd);
    m.def("indice_weighted_sum_bwd_input", &grid_sample::indice_weighted_sum_bwd_input);

    // Spconv (3 + 2 GEMM)
    m.def("hashmap_build_submanifold_conv_neighbour_map_cuda", &spconv::hashmap_build_submanifold_conv_neighbour_map_cuda);
    m.def("neighbor_map_post_process_for_masked_implicit_gemm_1", &spconv::neighbor_map_post_process_for_masked_implicit_gemm_1);
    m.def("neighbor_map_post_process_for_masked_implicit_gemm_2", &spconv::neighbor_map_post_process_for_masked_implicit_gemm_2);
    m.def("spconv_fwd_implicit_gemm", &spconv::spconv_fwd_implicit_gemm);
    m.def("spconv_bwd_implicit_gemm", &spconv::spconv_bwd_implicit_gemm);

    // Autotune timing cache query
    m.def("spconv_get_timing_cache", []() {
        std::lock_guard<std::mutex> lock(g_spconv_timing.mu);
        py::dict result;
        for (auto& [k, v] : g_spconv_timing.cache) {
            uint32_t nb  = (k >> 48) & 0xFFFF;
            uint32_t ci  = (k >> 32) & 0xFFFF;
            uint32_t co  = (k >> 16) & 0xFFFF;
            uint32_t vol = k & 0xFFFF;
            auto key_str = std::to_string(nb) + "x" + std::to_string(ci) + "x" +
                           std::to_string(co) + "x" + std::to_string(vol);
            result[py::cast(key_str)] = v;
        }
        return result;
    });
    m.def("spconv_clear_timing_cache", []() {
        std::lock_guard<std::mutex> lock(g_spconv_timing.mu);
        g_spconv_timing.cache.clear();
    });
}

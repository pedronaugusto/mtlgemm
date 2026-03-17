#import <torch/extension.h>
#import <Metal/Metal.h>
#import "common/metal_context.h"
#import "hash/api.h"
#import "serialize/api.h"
#import "grid_sample/api.h"
#import "spconv/api.h"

#include <dlfcn.h>

#define BLOCK_SIZE 256

// ============================================================================
// Buffer helpers — StorageModeShared MTLBuffers
// tensor → MTLBuffer → dispatch → tensor
// ============================================================================

namespace flex_gemm {
namespace metal {

static MetalContext& ctx() { return MetalContext::instance(); }

static id<MTLBuffer> alloc(size_t bytes) {
    return [ctx().device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> from_tensor(const torch::Tensor& t) {
    auto tc = t.contiguous().cpu();
    size_t bytes = tc.nbytes();
    auto buf = alloc(bytes);
    memcpy([buf contents], tc.data_ptr(), bytes);
    return buf;
}

static id<MTLBuffer> from_tensor_inplace(torch::Tensor& t) {
    // For in-place mutation (hashmap keys/values): tensor must be contiguous CPU
    TORCH_CHECK(t.is_contiguous(), "In-place tensor must be contiguous");
    auto tc = t.cpu();
    size_t bytes = tc.nbytes();
    auto buf = alloc(bytes);
    memcpy([buf contents], tc.data_ptr(), bytes);
    return buf;
}

static void to_tensor(id<MTLBuffer> buf, torch::Tensor& t) {
    // Copy MTLBuffer contents back into a CPU tensor
    memcpy(t.data_ptr(), [buf contents], t.nbytes());
}

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
        [enc setBuffer:buf_hk offset:0 atIndex:0];
        [enc setBuffer:buf_hv offset:0 atIndex:1];
        [enc setBuffer:buf_k offset:0 atIndex:2];
        [enc setBuffer:buf_v offset:0 atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&M length:sizeof(M) atIndex:5];
    }, M);

    to_tensor(buf_hk, hashmap_keys);
    to_tensor(buf_hv, hashmap_values);
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
    auto output = torch::empty({(int64_t)M}, torch::dtype(hashmap_values.dtype()));
    auto buf_out = alloc(output.nbytes());

    ctx().dispatch("hashmap_lookup_u32_u32", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_hk offset:0 atIndex:0];
        [enc setBuffer:buf_hv offset:0 atIndex:1];
        [enc setBuffer:buf_k offset:0 atIndex:2];
        [enc setBuffer:buf_out offset:0 atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&M length:sizeof(M) atIndex:5];
    }, M);

    to_tensor(buf_out, output);
    return output;
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
            [enc setBuffer:buf_hi offset:0 atIndex:0];
            [enc setBuffer:buf_lo offset:0 atIndex:1];
            [enc setBuffer:buf_hv offset:0 atIndex:2];
            [enc setBuffer:buf_coords offset:0 atIndex:3];
            [enc setBuffer:buf_vals offset:0 atIndex:4];
            [enc setBytes:&N length:sizeof(N) atIndex:5];
            [enc setBytes:&M length:sizeof(M) atIndex:6];
            [enc setBytes:&W length:sizeof(W) atIndex:7];
            [enc setBytes:&H length:sizeof(H) atIndex:8];
            [enc setBytes:&D length:sizeof(D) atIndex:9];
        }, M);

        to_tensor(buf_hv, hashmap_values);
        // Rejoin hi/lo back into uint64 tensor
        to_tensor(buf_hi, keys_hi);
        to_tensor(buf_lo, keys_lo);
        auto rejoined = torch::empty({(int64_t)N * 2}, torch::kUInt32);
        // Interleave: lo at even, hi at odd
        for (uint32_t i = 0; i < N; i++) {
            ((uint32_t*)rejoined.data_ptr())[2*i]     = ((uint32_t*)keys_lo.data_ptr())[i];
            ((uint32_t*)rejoined.data_ptr())[2*i + 1] = ((uint32_t*)keys_hi.data_ptr())[i];
        }
        memcpy(hashmap_keys.data_ptr(), rejoined.data_ptr(), hashmap_keys.nbytes());
    } else {
        TORCH_CHECK(hashmap_keys.dtype() == torch::kUInt32, "Keys must be uint32 or uint64");
        uint32_t N = (uint32_t)hashmap_keys.size(0);

        auto buf_hk = from_tensor_inplace(hashmap_keys);
        auto buf_hv = from_tensor_inplace(hashmap_values);

        ctx().dispatch("hashmap_insert_3d_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk offset:0 atIndex:0];
            [enc setBuffer:buf_hv offset:0 atIndex:1];
            [enc setBuffer:buf_coords offset:0 atIndex:2];
            [enc setBuffer:buf_vals offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&M length:sizeof(M) atIndex:5];
            [enc setBytes:&W length:sizeof(W) atIndex:6];
            [enc setBytes:&H length:sizeof(H) atIndex:7];
            [enc setBytes:&D length:sizeof(D) atIndex:8];
        }, M);

        to_tensor(buf_hk, hashmap_keys);
        to_tensor(buf_hv, hashmap_values);
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
    auto output = torch::empty({(int64_t)M}, torch::dtype(hashmap_values.dtype()));
    auto buf_out = alloc(output.nbytes());

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
            [enc setBuffer:buf_hi offset:0 atIndex:0];
            [enc setBuffer:buf_lo offset:0 atIndex:1];
            [enc setBuffer:buf_hv offset:0 atIndex:2];
            [enc setBuffer:buf_coords offset:0 atIndex:3];
            [enc setBuffer:buf_out offset:0 atIndex:4];
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
            [enc setBuffer:buf_hk offset:0 atIndex:0];
            [enc setBuffer:buf_hv offset:0 atIndex:1];
            [enc setBuffer:buf_coords offset:0 atIndex:2];
            [enc setBuffer:buf_out offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&M length:sizeof(M) atIndex:5];
            [enc setBytes:&W length:sizeof(W) atIndex:6];
            [enc setBytes:&H length:sizeof(H) atIndex:7];
            [enc setBytes:&D length:sizeof(D) atIndex:8];
        }, M);
    }

    to_tensor(buf_out, output);
    return output;
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
            [enc setBuffer:buf_hi offset:0 atIndex:0];
            [enc setBuffer:buf_lo offset:0 atIndex:1];
            [enc setBuffer:buf_hv offset:0 atIndex:2];
            [enc setBuffer:buf_coords offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&M length:sizeof(M) atIndex:5];
            [enc setBytes:&W length:sizeof(W) atIndex:6];
            [enc setBytes:&H length:sizeof(H) atIndex:7];
            [enc setBytes:&D length:sizeof(D) atIndex:8];
        }, M);

        to_tensor(buf_hv, hashmap_values);
        to_tensor(buf_hi, keys_hi);
        to_tensor(buf_lo, keys_lo);
        auto rejoined = torch::empty({(int64_t)N * 2}, torch::kUInt32);
        for (uint32_t i = 0; i < N; i++) {
            ((uint32_t*)rejoined.data_ptr())[2*i]     = ((uint32_t*)keys_lo.data_ptr())[i];
            ((uint32_t*)rejoined.data_ptr())[2*i + 1] = ((uint32_t*)keys_hi.data_ptr())[i];
        }
        memcpy(hashmap_keys.data_ptr(), rejoined.data_ptr(), hashmap_keys.nbytes());
    } else {
        TORCH_CHECK(hashmap_keys.dtype() == torch::kUInt32, "Keys must be uint32 or uint64");
        uint32_t N = (uint32_t)hashmap_keys.size(0);

        auto buf_hk = from_tensor_inplace(hashmap_keys);
        auto buf_hv = from_tensor_inplace(hashmap_values);

        ctx().dispatch("hashmap_insert_3d_idx_as_val_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk offset:0 atIndex:0];
            [enc setBuffer:buf_hv offset:0 atIndex:1];
            [enc setBuffer:buf_coords offset:0 atIndex:2];
            [enc setBytes:&N length:sizeof(N) atIndex:3];
            [enc setBytes:&M length:sizeof(M) atIndex:4];
            [enc setBytes:&W length:sizeof(W) atIndex:5];
            [enc setBytes:&H length:sizeof(H) atIndex:6];
            [enc setBytes:&D length:sizeof(D) atIndex:7];
        }, M);

        to_tensor(buf_hk, hashmap_keys);
        to_tensor(buf_hv, hashmap_values);
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
    auto buf_codes = alloc(codes.nbytes());

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "z_order_encode_3d_u32" : "z_order_encode_3d_u64";

    ctx().dispatch(kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_coords offset:0 atIndex:0];
        [enc setBuffer:buf_codes offset:0 atIndex:1];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:2];
        [enc setBytes:&bl length:sizeof(bl) atIndex:3];
    }, N_val);

    to_tensor(buf_codes, codes);
}

torch::Tensor z_order_decode(
    const torch::Tensor& codes,
    const size_t bit_length
) {
    auto codes_c = codes.cpu().contiguous();
    auto result_coords = torch::empty({codes.size(0), 4}, torch::kInt32);
    uint32_t N_val = (uint32_t)codes.size(0);
    uint32_t bl = (uint32_t)bit_length;

    auto buf_codes = from_tensor(codes_c);
    auto buf_coords = alloc(result_coords.nbytes());

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "z_order_decode_3d_u32" : "z_order_decode_3d_u64";

    ctx().dispatch(kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_codes offset:0 atIndex:0];
        [enc setBuffer:buf_coords offset:0 atIndex:1];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:2];
        [enc setBytes:&bl length:sizeof(bl) atIndex:3];
    }, N_val);

    to_tensor(buf_coords, result_coords);
    return result_coords;
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
    auto buf_codes = alloc(codes.nbytes());

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "hilbert_encode_3d_u32" : "hilbert_encode_3d_u64";

    ctx().dispatch(kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_coords offset:0 atIndex:0];
        [enc setBuffer:buf_codes offset:0 atIndex:1];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:2];
        [enc setBytes:&bl length:sizeof(bl) atIndex:3];
    }, N_val);

    to_tensor(buf_codes, codes);
}

torch::Tensor hilbert_decode(
    const torch::Tensor& codes,
    const size_t bit_length
) {
    auto codes_c = codes.cpu().contiguous();
    auto result_coords = torch::empty({codes.size(0), 4}, torch::kInt32);
    uint32_t N_val = (uint32_t)codes.size(0);
    uint32_t bl = (uint32_t)bit_length;

    auto buf_codes = from_tensor(codes_c);
    auto buf_coords = alloc(result_coords.nbytes());

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "hilbert_decode_3d_u32" : "hilbert_decode_3d_u64";

    ctx().dispatch(kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_codes offset:0 atIndex:0];
        [enc setBuffer:buf_coords offset:0 atIndex:1];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:2];
        [enc setBytes:&bl length:sizeof(bl) atIndex:3];
    }, N_val);

    to_tensor(buf_coords, result_coords);
    return result_coords;
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

    auto neighbor = torch::full({(int64_t)B, (int64_t)L}, (int64_t)0xFFFFFFFF, torch::kUInt32);

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_vals);
    auto buf_grid = from_tensor(grid);
    auto buf_neigh = from_tensor_inplace(neighbor);

    if (hashmap_keys.dtype() == torch::kUInt64) {
        auto keys_flat = hashmap_keys.contiguous().view({-1});
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)N * 2}, torch::kUInt32).clone();
        auto klo = keys_u32.slice(0, 0, N * 2, 2).contiguous();
        auto khi = keys_u32.slice(0, 1, N * 2, 2).contiguous();
        auto buf_hi = from_tensor(khi);
        auto buf_lo = from_tensor(klo);

        ctx().dispatch("grid_sample_nearest_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi offset:0 atIndex:0];
            [enc setBuffer:buf_lo offset:0 atIndex:1];
            [enc setBuffer:buf_hv offset:0 atIndex:2];
            [enc setBuffer:buf_grid offset:0 atIndex:3];
            [enc setBuffer:buf_neigh offset:0 atIndex:4];
            [enc setBytes:&N length:sizeof(N) atIndex:5];
            [enc setBytes:&B length:sizeof(B) atIndex:6];
            [enc setBytes:&L length:sizeof(L) atIndex:7];
            [enc setBytes:&W length:sizeof(W) atIndex:8];
            [enc setBytes:&H length:sizeof(H) atIndex:9];
            [enc setBytes:&D length:sizeof(D) atIndex:10];
        }, B * L);
    } else {
        ctx().dispatch("grid_sample_nearest_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk offset:0 atIndex:0];
            [enc setBuffer:buf_hv offset:0 atIndex:1];
            [enc setBuffer:buf_grid offset:0 atIndex:2];
            [enc setBuffer:buf_neigh offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&B length:sizeof(B) atIndex:5];
            [enc setBytes:&L length:sizeof(L) atIndex:6];
            [enc setBytes:&W length:sizeof(W) atIndex:7];
            [enc setBytes:&H length:sizeof(H) atIndex:8];
            [enc setBytes:&D length:sizeof(D) atIndex:9];
        }, B * L);
    }

    to_tensor(buf_neigh, neighbor);
    return neighbor;
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

    auto neighbor = torch::full({(int64_t)B, (int64_t)L, 8}, (int64_t)0xFFFFFFFF, torch::kUInt32);
    auto weight = torch::zeros({(int64_t)B, (int64_t)L, 8}, torch::kFloat32);

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_vals);
    auto buf_grid = from_tensor(grid);
    auto buf_neigh = alloc(neighbor.nbytes());
    memcpy([buf_neigh contents], neighbor.data_ptr(), neighbor.nbytes());
    auto buf_weight = alloc(weight.nbytes());
    memset([buf_weight contents], 0, weight.nbytes());

    if (hashmap_keys.dtype() == torch::kUInt64) {
        auto keys_flat = hashmap_keys.contiguous().view({-1});
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)N * 2}, torch::kUInt32).clone();
        auto klo = keys_u32.slice(0, 0, N * 2, 2).contiguous();
        auto khi = keys_u32.slice(0, 1, N * 2, 2).contiguous();
        auto buf_hi = from_tensor(khi);
        auto buf_lo = from_tensor(klo);

        ctx().dispatch("grid_sample_trilinear_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi offset:0 atIndex:0];
            [enc setBuffer:buf_lo offset:0 atIndex:1];
            [enc setBuffer:buf_hv offset:0 atIndex:2];
            [enc setBuffer:buf_grid offset:0 atIndex:3];
            [enc setBuffer:buf_neigh offset:0 atIndex:4];
            [enc setBuffer:buf_weight offset:0 atIndex:5];
            [enc setBytes:&N length:sizeof(N) atIndex:6];
            [enc setBytes:&B length:sizeof(B) atIndex:7];
            [enc setBytes:&L length:sizeof(L) atIndex:8];
            [enc setBytes:&W length:sizeof(W) atIndex:9];
            [enc setBytes:&H length:sizeof(H) atIndex:10];
            [enc setBytes:&D length:sizeof(D) atIndex:11];
        }, B * L);
    } else {
        ctx().dispatch("grid_sample_trilinear_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk offset:0 atIndex:0];
            [enc setBuffer:buf_hv offset:0 atIndex:1];
            [enc setBuffer:buf_grid offset:0 atIndex:2];
            [enc setBuffer:buf_neigh offset:0 atIndex:3];
            [enc setBuffer:buf_weight offset:0 atIndex:4];
            [enc setBytes:&N length:sizeof(N) atIndex:5];
            [enc setBytes:&B length:sizeof(B) atIndex:6];
            [enc setBytes:&L length:sizeof(L) atIndex:7];
            [enc setBytes:&W length:sizeof(W) atIndex:8];
            [enc setBytes:&H length:sizeof(H) atIndex:9];
            [enc setBytes:&D length:sizeof(D) atIndex:10];
        }, B * L);
    }

    to_tensor(buf_neigh, neighbor);
    to_tensor(buf_weight, weight);
    return std::make_tuple(neighbor, weight);
}

torch::Tensor indice_weighted_sum_fwd(
    const torch::Tensor& input,
    const torch::Tensor& indices,
    const torch::Tensor& weight
) {
    uint32_t M = (uint32_t)indices.size(0);
    uint32_t C = (uint32_t)input.size(1);
    uint32_t V = (uint32_t)weight.size(1);

    auto output = torch::empty({(int64_t)M, (int64_t)C}, input.dtype());

    auto buf_in = from_tensor(input);
    auto buf_idx = from_tensor(indices);
    auto buf_w = from_tensor(weight);
    auto buf_out = alloc(output.nbytes());

    uint32_t tg_x = std::min(C, (uint32_t)32);
    uint32_t tg_y = std::min(M, (uint32_t)(256 / tg_x));
    MTLSize grid = MTLSizeMake((C + tg_x - 1) / tg_x, (M + tg_y - 1) / tg_y, 1);
    MTLSize threadgroup = MTLSizeMake(tg_x, tg_y, 1);

    ctx().dispatch_2d("indice_weighted_sum_fwd", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_in offset:0 atIndex:0];
        [enc setBuffer:buf_idx offset:0 atIndex:1];
        [enc setBuffer:buf_w offset:0 atIndex:2];
        [enc setBuffer:buf_out offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(M) atIndex:4];
        [enc setBytes:&C length:sizeof(C) atIndex:5];
        [enc setBytes:&V length:sizeof(V) atIndex:6];
    }, grid, threadgroup);

    to_tensor(buf_out, output);
    return output;
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

    auto grad_input = torch::zeros({N, (int64_t)C}, grad_output.dtype());

    auto buf_go = from_tensor(grad_output);
    auto buf_idx = from_tensor(indices);
    auto buf_w = from_tensor(weight);
    auto buf_gi = alloc(grad_input.nbytes());
    memset([buf_gi contents], 0, grad_input.nbytes());

    uint32_t MV = M * V;
    uint32_t tg_x = std::min(C, (uint32_t)32);
    uint32_t tg_y = std::min(MV, (uint32_t)(256 / tg_x));
    MTLSize grid = MTLSizeMake((C + tg_x - 1) / tg_x, (MV + tg_y - 1) / tg_y, 1);
    MTLSize threadgroup = MTLSizeMake(tg_x, tg_y, 1);

    ctx().dispatch_2d("indice_weighted_sum_bwd_input", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_go offset:0 atIndex:0];
        [enc setBuffer:buf_idx offset:0 atIndex:1];
        [enc setBuffer:buf_w offset:0 atIndex:2];
        [enc setBuffer:buf_gi offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(M) atIndex:4];
        [enc setBytes:&C length:sizeof(C) atIndex:5];
        [enc setBytes:&V length:sizeof(V) atIndex:6];
    }, grid, threadgroup);

    to_tensor(buf_gi, grad_input);
    return grad_input;
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

    auto neighbor = torch::full({coords.size(0), (int64_t)V}, (int64_t)0xFFFFFFFF, torch::kUInt32);

    uint32_t hash_N = (uint32_t)hashmap_keys.size(0);
    uint32_t M = (uint32_t)coords.size(0);
    uint64_t thread_count = (uint64_t)M * (V / 2 + 1);

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_vals);
    auto buf_coords = from_tensor(coords);
    auto buf_neigh = alloc(neighbor.nbytes());
    memcpy([buf_neigh contents], neighbor.data_ptr(), neighbor.nbytes());

    if (hashmap_keys.dtype() == torch::kUInt64) {
        auto keys_flat = hashmap_keys.contiguous().view({-1});
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)hash_N * 2}, torch::kUInt32).clone();
        auto klo = keys_u32.slice(0, 0, hash_N * 2, 2).contiguous();
        auto khi = keys_u32.slice(0, 1, hash_N * 2, 2).contiguous();
        auto buf_hi = from_tensor(khi);
        auto buf_lo = from_tensor(klo);

        ctx().dispatch("submanifold_conv_neighbor_map_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi offset:0 atIndex:0];
            [enc setBuffer:buf_lo offset:0 atIndex:1];
            [enc setBuffer:buf_hv offset:0 atIndex:2];
            [enc setBuffer:buf_coords offset:0 atIndex:3];
            [enc setBuffer:buf_neigh offset:0 atIndex:4];
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
            [enc setBuffer:buf_hk offset:0 atIndex:0];
            [enc setBuffer:buf_hv offset:0 atIndex:1];
            [enc setBuffer:buf_coords offset:0 atIndex:2];
            [enc setBuffer:buf_neigh offset:0 atIndex:3];
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

    to_tensor(buf_neigh, neighbor);
    return neighbor;
    } // @autoreleasepool
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
neighbor_map_post_process_for_masked_implicit_gemm_1(
    const torch::Tensor& neighbor_map
) {
    @autoreleasepool {
    int64_t N = neighbor_map.size(0);
    int64_t V = neighbor_map.size(1);

    auto gray_code = torch::empty({N}, torch::kInt32);
    auto binary_code = torch::empty({N}, torch::kInt32);
    auto neigh_mask_T = torch::empty({V * N}, torch::kInt32);
    auto neigh_map_T = torch::empty({V * N}, torch::kUInt32);

    uint32_t N32 = (uint32_t)N;
    uint32_t V32 = (uint32_t)V;
    uint32_t tg_size = 256;
    uint32_t num_groups = (N32 + tg_size - 1) / tg_size;
    uint32_t shared_mem = tg_size * V32 * sizeof(uint32_t);

    auto buf_nm = from_tensor(neighbor_map);
    auto buf_gc = alloc(gray_code.nbytes());
    auto buf_bc = alloc(binary_code.nbytes());
    auto buf_nmt = alloc(neigh_map_T.nbytes());
    auto buf_nmaskt = alloc(neigh_mask_T.nbytes());

    auto pso = ctx().pipeline("neighbor_map_gray_code");

    {
        id<MTLCommandBuffer> cmdbuf = [ctx().queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_nm offset:0 atIndex:0];
        [enc setBuffer:buf_gc offset:0 atIndex:1];
        [enc setBuffer:buf_bc offset:0 atIndex:2];
        [enc setBuffer:buf_nmt offset:0 atIndex:3];
        [enc setBuffer:buf_nmaskt offset:0 atIndex:4];
        [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
        [enc setBytes:&V32 length:sizeof(V32) atIndex:6];
        [enc setThreadgroupMemoryLength:shared_mem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }

    to_tensor(buf_gc, gray_code);
    to_tensor(buf_bc, binary_code);
    to_tensor(buf_nmt, neigh_map_T);
    to_tensor(buf_nmaskt, neigh_mask_T);

    auto sorted_idx = torch::argsort(binary_code);

    // Prefix sum and gather
    auto prefix_sum = torch::cumsum(neigh_mask_T, 0, torch::kInt32);
    auto total_valid = prefix_sum[-1].item<int32_t>();
    auto valid_signal_i = torch::empty({total_valid}, torch::kUInt32);
    auto valid_signal_o = torch::empty({total_valid}, torch::kUInt32);
    auto valid_signal_seg = torch::empty({V + 1}, torch::kUInt32);

    auto buf_ps = from_tensor(prefix_sum);
    auto buf_nmt2 = from_tensor(neigh_map_T);
    auto buf_vso = alloc(valid_signal_o.nbytes());
    auto buf_vsi = alloc(valid_signal_i.nbytes());
    auto buf_vss = alloc(valid_signal_seg.nbytes());

    ctx().dispatch("gather_idx_val_seg_from_prefix_sum", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_ps offset:0 atIndex:0];
        [enc setBuffer:buf_nmt2 offset:0 atIndex:1];
        [enc setBuffer:buf_vso offset:0 atIndex:2];
        [enc setBuffer:buf_vsi offset:0 atIndex:3];
        [enc setBuffer:buf_vss offset:0 atIndex:4];
        [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
        [enc setBytes:&V32 length:sizeof(V32) atIndex:6];
    }, N * V);

    to_tensor(buf_vsi, valid_signal_i);
    to_tensor(buf_vso, valid_signal_o);
    to_tensor(buf_vss, valid_signal_seg);

    return std::make_tuple(gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg);
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

    auto reduced_code = torch::empty({num_blocks}, torch::kInt32);
    auto seglen = torch::empty({num_blocks + 1}, torch::kInt32);

    int block_dim = block_size;
    uint32_t tg_size = 256;
    uint32_t num_dispatch_groups = ((N + 1) / 2 + tg_size - 1) / tg_size;
    uint32_t shared_mem = tg_size * sizeof(int32_t);

    auto buf_gc = from_tensor(gray_code);
    auto buf_si = from_tensor(sorted_idx);
    auto buf_rc = alloc(reduced_code.nbytes());
    auto buf_sl = alloc(seglen.nbytes());

    auto pso = ctx().pipeline("reduce_code");

    {
        id<MTLCommandBuffer> cmdbuf = [ctx().queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_gc offset:0 atIndex:0];
        [enc setBuffer:buf_si offset:0 atIndex:1];
        [enc setBuffer:buf_rc offset:0 atIndex:2];
        [enc setBuffer:buf_sl offset:0 atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&block_dim length:sizeof(block_dim) atIndex:5];
        [enc setThreadgroupMemoryLength:shared_mem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(num_dispatch_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }

    to_tensor(buf_sl, seglen);
    seglen = torch::cumsum(seglen, 0, torch::kInt32);

    auto total_valid = seglen[-1].item<int32_t>();
    auto valid_kernel_idx = torch::empty({total_valid}, torch::kInt32);
    uint32_t nb = (uint32_t)num_blocks;

    to_tensor(buf_rc, reduced_code);
    auto buf_sl2 = from_tensor(seglen);
    auto buf_rc2 = from_tensor(reduced_code);
    auto buf_vki = alloc(valid_kernel_idx.nbytes());

    ctx().dispatch("scatter_reduced_code", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_rc2 offset:0 atIndex:0];
        [enc setBuffer:buf_sl2 offset:0 atIndex:1];
        [enc setBuffer:buf_vki offset:0 atIndex:2];
        [enc setBytes:&nb length:sizeof(nb) atIndex:3];
    }, num_blocks);

    to_tensor(buf_vki, valid_kernel_idx);
    return std::make_tuple(valid_kernel_idx, seglen);
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

    auto output = torch::empty({(int64_t)N, (int64_t)Co}, input.dtype());

    auto buf_input    = from_tensor(input);
    auto buf_weight   = from_tensor(weight);
    auto buf_bias     = (bias.numel() > 0) ? from_tensor(bias) : alloc(4);
    auto buf_neighbor = from_tensor(neighbor);
    auto buf_output   = alloc(output.nbytes());

    uint32_t has_bias = (bias.numel() > 0) ? 1 : 0;

    // Grid: (cdiv(N, 32), cdiv(Co, 32)), Threadgroup: 64
    uint32_t grid_x = (N + 31) / 32;
    uint32_t grid_y = (Co + 31) / 32;
    // Shared memory: B1*BK + BK*B2 + B1 (as uint) = 32*32 + 32*32 + 32 = 2080 floats
    // But smem_nb is uint, reinterpreted from float — 32 uints = 32 floats worth
    uint32_t shared_mem = (32 * 32 + 32 * 32 + 32) * sizeof(float);

    auto pso = ctx().pipeline("spconv_fwd_implicit_gemm");

    @autoreleasepool {
        id<MTLCommandBuffer> cmdbuf = [ctx().queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf_input offset:0 atIndex:0];
        [enc setBuffer:buf_weight offset:0 atIndex:1];
        [enc setBuffer:buf_bias offset:0 atIndex:2];
        [enc setBuffer:buf_neighbor offset:0 atIndex:3];
        [enc setBuffer:buf_output offset:0 atIndex:4];
        [enc setBytes:&N length:sizeof(N) atIndex:5];
        [enc setBytes:&Co length:sizeof(Co) atIndex:6];
        [enc setBytes:&Ci length:sizeof(Ci) atIndex:7];
        [enc setBytes:&V length:sizeof(V) atIndex:8];
        [enc setBytes:&has_bias length:sizeof(has_bias) atIndex:9];
        [enc setThreadgroupMemoryLength:shared_mem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }

    to_tensor(buf_output, output);
    return output;
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

    auto grad_input  = torch::empty({(int64_t)N, (int64_t)Ci}, input.dtype());
    auto grad_weight = torch::empty({(int64_t)Co, (int64_t)V, (int64_t)Ci}, weight.dtype());

    auto buf_go       = from_tensor(grad_output);
    auto buf_input    = from_tensor(input);
    auto buf_weight   = from_tensor(weight);
    auto buf_neighbor = from_tensor(neighbor);
    auto buf_gi       = alloc(grad_input.nbytes());
    auto buf_gw       = alloc(grad_weight.nbytes());

    // Shared memory for bwd_input: B1*BK + BK*B2 + B1 (smem_nb) = 2080 floats
    uint32_t shared_mem_input = (32 * 32 + 32 * 32 + 32) * sizeof(float);
    // Shared memory for bwd_weight: B1*BK + BK*B2 = 2048 floats
    uint32_t shared_mem_weight = (32 * 32 + 32 * 32) * sizeof(float);

    auto pso_gi = ctx().pipeline("spconv_bwd_input_implicit_gemm");
    auto pso_gw = ctx().pipeline("spconv_bwd_weight_implicit_gemm");

    @autoreleasepool {
        id<MTLCommandBuffer> cmdbuf = [ctx().queue() commandBuffer];

        // Backward-input kernel
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:pso_gi];
            [enc setBuffer:buf_go offset:0 atIndex:0];
            [enc setBuffer:buf_weight offset:0 atIndex:1];
            [enc setBuffer:buf_neighbor offset:0 atIndex:2];
            [enc setBuffer:buf_gi offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&Co length:sizeof(Co) atIndex:5];
            [enc setBytes:&Ci length:sizeof(Ci) atIndex:6];
            [enc setBytes:&V length:sizeof(V) atIndex:7];
            [enc setThreadgroupMemoryLength:shared_mem_input atIndex:0];
            uint32_t grid_x = (N + 31) / 32;
            uint32_t grid_y = (Ci + 31) / 32;
            [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }

        // Backward-weight kernel
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:pso_gw];
            [enc setBuffer:buf_go offset:0 atIndex:0];
            [enc setBuffer:buf_input offset:0 atIndex:1];
            [enc setBuffer:buf_neighbor offset:0 atIndex:2];
            [enc setBuffer:buf_gw offset:0 atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&Co length:sizeof(Co) atIndex:5];
            [enc setBytes:&Ci length:sizeof(Ci) atIndex:6];
            [enc setBytes:&V length:sizeof(V) atIndex:7];
            [enc setThreadgroupMemoryLength:shared_mem_weight atIndex:0];
            uint32_t grid_x = (Co + 31) / 32;
            uint32_t grid_y = (VCi + 31) / 32;
            [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }

        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }

    to_tensor(buf_gi, grad_input);
    to_tensor(buf_gw, grad_weight);
    return std::make_tuple(grad_input, grad_weight);
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
}

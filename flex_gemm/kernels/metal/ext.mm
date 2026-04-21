#import <torch/extension.h>
#import <Metal/Metal.h>
#import <ATen/mps/MPSStream.h>
#import "common/metal_context.h"

// Defined in <ATen/native/mps/OperationUtils.h>, but that header pulls in
// MPS graph headers that use retain/release and don't compile under ARC.
// The function is a one-line bit-cast of the storage pointer to MTLBuffer,
// so we forward-declare and inline it ourselves.
namespace at { namespace native { namespace mps {
static inline id<MTLBuffer> getMTLBufferStorage(const at::TensorBase& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}
}}}

#import "hash/api.h"
#import "serialize/api.h"
#import "grid_sample/api.h"
#import "spconv/api.h"

#include <dlfcn.h>
#include <chrono>
#include <mutex>

#define BLOCK_SIZE 256

// ============================================================================
// Buffer helpers — device-aware zero-copy MTLBuffers.
//
// CPU tensors:  wrapped via newBufferWithBytesNoCopy on shared storage.
// MPS tensors:  wrapped via at::native::mps::getMTLBufferStorage on the
//               PyTorch-allocated MTLBuffer directly. Output tensors are
//               allocated on the same device as the input, and dispatch
//               is routed through the appropriate command queue.
// ============================================================================

namespace flex_gemm {
namespace metal {

static MetalContext& ctx() { return MetalContext::instance(); }

static id<MTLBuffer> alloc(size_t bytes) {
    return [ctx().device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

// Wraps a tensor as an MTLBuffer for kernel dispatch.
// `backing` keeps the tensor alive for the duration of the dispatch.
// `offset` accounts for non-zero storage_offset on MPS tensors.
// `is_mps` flags whether dispatch must go through PyTorch's MPSStream.
struct TensorBuffer {
    id<MTLBuffer> buffer;
    NSUInteger offset;
    torch::Tensor backing;
    bool is_mps;
};

static TensorBuffer from_tensor(const torch::Tensor& t) {
    TORCH_CHECK(t.numel() > 0, "Cannot wrap empty tensor as Metal buffer");
    if (t.device().is_mps()) {
        auto tc = t.contiguous();  // contiguous() on MPS stays on MPS
        id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(tc);
        TORCH_CHECK(buf != nil, "Failed to obtain MTLBuffer from MPS tensor");
        NSUInteger offset = (NSUInteger)tc.storage_offset() * tc.element_size();
        return {buf, offset, tc, true};
    } else {
        auto tc = t.contiguous().cpu();
        id<MTLBuffer> buf = [ctx().device() newBufferWithBytesNoCopy:tc.data_ptr()
                                                               length:tc.nbytes()
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        TORCH_CHECK(buf != nil, "Failed to wrap CPU tensor as Metal buffer");
        return {buf, 0, tc, false};
    }
}

// In-place wrap. Caller mutates the original tensor through the returned buffer.
// On CPU we may need to reassign the caller's tensor variable to the cpu()'d
// version if we forced a host copy — that's the caller's responsibility via
// the returned `backing` field. On MPS the wrap is true zero-copy.
static TensorBuffer from_tensor_inplace(torch::Tensor& t) {
    TORCH_CHECK(t.is_contiguous(), "In-place tensor must be contiguous");
    TORCH_CHECK(t.numel() > 0, "Cannot wrap empty tensor as Metal buffer");
    if (t.device().is_mps()) {
        id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(t);
        TORCH_CHECK(buf != nil, "Failed to obtain MTLBuffer from MPS tensor");
        NSUInteger offset = (NSUInteger)t.storage_offset() * t.element_size();
        return {buf, offset, t, true};
    } else {
        auto tc = t.cpu();
        id<MTLBuffer> buf = [ctx().device() newBufferWithBytesNoCopy:tc.data_ptr()
                                                               length:tc.nbytes()
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        TORCH_CHECK(buf != nil, "Failed to wrap CPU tensor as Metal buffer");
        return {buf, 0, tc, false};
    }
}

// Output allocator: produces a tensor on `device`, zero-copy wrapped as MTLBuffer.
static TensorBuffer make_output(const std::vector<int64_t>& sizes,
                                torch::ScalarType dtype,
                                c10::Device device) {
    auto opts = torch::TensorOptions().dtype(dtype).device(device);
    auto t = torch::empty(sizes, opts);
    TORCH_CHECK(t.numel() > 0, "Cannot create Metal buffer from empty tensor");
    if (device.is_mps()) {
        id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(t);
        TORCH_CHECK(buf != nil, "Failed to obtain MTLBuffer from MPS output tensor");
        NSUInteger offset = (NSUInteger)t.storage_offset() * t.element_size();
        return {buf, offset, t, true};
    } else {
        id<MTLBuffer> buf = [ctx().device() newBufferWithBytesNoCopy:t.data_ptr()
                                                               length:t.nbytes()
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        TORCH_CHECK(buf != nil, "Failed to wrap CPU output tensor as Metal buffer");
        return {buf, 0, t, false};
    }
}

// Whether torch.zeros has an MPS kernel for this dtype (today it does for fp,
// not for int). Used to decide whether to stage through CPU.
static bool mps_zeros_supported(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32:
        case torch::kFloat16:
        case torch::kBFloat16:
            return true;
        default:
            return false;
    }
}

static TensorBuffer make_output_zeroed(const std::vector<int64_t>& sizes,
                                       torch::ScalarType dtype,
                                       c10::Device device) {
    TORCH_CHECK(c10::multiply_integers(sizes) > 0, "Cannot create Metal buffer from empty tensor");
    if (device.is_mps()) {
        torch::Tensor t;
        if (mps_zeros_supported(dtype)) {
            t = torch::zeros(sizes, torch::TensorOptions().dtype(dtype).device(device));
        } else {
            // Stage through CPU for int dtypes — torch.zeros lacks an MPS kernel.
            auto t_cpu = torch::zeros(sizes, torch::TensorOptions().dtype(dtype));
            t = t_cpu.to(device);
        }
        id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(t);
        TORCH_CHECK(buf != nil, "Failed to obtain MTLBuffer from MPS output tensor");
        NSUInteger offset = (NSUInteger)t.storage_offset() * t.element_size();
        return {buf, offset, t, true};
    } else {
        auto t = torch::zeros(sizes, torch::TensorOptions().dtype(dtype).device(device));
        id<MTLBuffer> buf = [ctx().device() newBufferWithBytesNoCopy:t.data_ptr()
                                                               length:t.nbytes()
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        TORCH_CHECK(buf != nil, "Failed to wrap CPU output tensor as Metal buffer");
        return {buf, 0, t, false};
    }
}

static TensorBuffer make_output_filled(const std::vector<int64_t>& sizes,
                                       torch::ScalarType dtype,
                                       int64_t fill_val,
                                       c10::Device device) {
    if (device.is_mps()) {
        // PyTorch MPS doesn't have a torch.full kernel for uint32 today, so
        // stage via CPU and transfer. Allocation is one-time per shape so the
        // copy cost is negligible against the per-step kernel time.
        auto t_cpu = torch::full(sizes, fill_val, torch::TensorOptions().dtype(dtype));
        auto t = t_cpu.to(device);
        id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(t);
        TORCH_CHECK(buf != nil, "Failed to obtain MTLBuffer from MPS output tensor");
        NSUInteger offset = (NSUInteger)t.storage_offset() * t.element_size();
        return {buf, offset, t, true};
    } else {
        auto t = torch::full(sizes, fill_val, torch::TensorOptions().dtype(dtype).device(device));
        id<MTLBuffer> buf = [ctx().device() newBufferWithBytesNoCopy:t.data_ptr()
                                                               length:t.nbytes()
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        TORCH_CHECK(buf != nil, "Failed to wrap CPU output tensor as Metal buffer");
        return {buf, 0, t, false};
    }
}

// Convenience: pick the right dispatch flavor based on whether the input is MPS.
static inline void dispatch_auto(
    bool on_mps,
    const std::string& kernel_name,
    std::function<void(id<MTLComputeCommandEncoder>)> setup,
    uint64_t thread_count
) {
    if (on_mps) {
        ctx().dispatch_mps(kernel_name, setup, thread_count);
    } else {
        ctx().dispatch(kernel_name, setup, thread_count);
    }
}

static inline void dispatch_2d_auto(
    bool on_mps,
    const std::string& kernel_name,
    std::function<void(id<MTLComputeCommandEncoder>)> setup,
    MTLSize grid_size,
    MTLSize threadgroup_size
) {
    if (on_mps) {
        ctx().dispatch_2d_mps(kernel_name, setup, grid_size, threadgroup_size);
    } else {
        ctx().dispatch_2d(kernel_name, setup, grid_size, threadgroup_size);
    }
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

    bool on_mps = hashmap_keys.device().is_mps();
    if (!on_mps) {
        hashmap_keys = hashmap_keys.cpu().contiguous();
        hashmap_values = hashmap_values.cpu().contiguous();
    } else {
        hashmap_keys = hashmap_keys.contiguous();
        hashmap_values = hashmap_values.contiguous();
    }

    uint32_t N = (uint32_t)hashmap_keys.size(0);
    uint32_t M = (uint32_t)keys.size(0);

    auto buf_hk = from_tensor_inplace(hashmap_keys);
    auto buf_hv = from_tensor_inplace(hashmap_values);
    auto buf_k = from_tensor(keys);
    auto buf_v = from_tensor(values);

    dispatch_auto(on_mps, "hashmap_insert_u32", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_hk.buffer offset:buf_hk.offset atIndex:0];
        [enc setBuffer:buf_hv.buffer offset:buf_hv.offset atIndex:1];
        [enc setBuffer:buf_k.buffer  offset:buf_k.offset  atIndex:2];
        [enc setBuffer:buf_v.buffer  offset:buf_v.offset  atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&M length:sizeof(M) atIndex:5];
    }, M);

    if (!on_mps) {
        if (hashmap_keys.data_ptr() != buf_hk.backing.data_ptr())
            memcpy(hashmap_keys.data_ptr(), buf_hk.backing.data_ptr(), hashmap_keys.nbytes());
        if (hashmap_values.data_ptr() != buf_hv.backing.data_ptr())
            memcpy(hashmap_values.data_ptr(), buf_hv.backing.data_ptr(), hashmap_values.nbytes());
    }
}

torch::Tensor hashmap_lookup_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& keys
) {
    TORCH_CHECK(hashmap_keys.dtype() == torch::kUInt32, "Only uint32 keys supported");

    bool on_mps = keys.device().is_mps();
    uint32_t N = (uint32_t)hashmap_keys.size(0);
    uint32_t M = (uint32_t)keys.size(0);

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_values);
    auto buf_k = from_tensor(keys);
    auto out = make_output({(int64_t)M}, hashmap_values.scalar_type(), keys.device());

    dispatch_auto(on_mps, "hashmap_lookup_u32_u32", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_hk.buffer offset:buf_hk.offset atIndex:0];
        [enc setBuffer:buf_hv.buffer offset:buf_hv.offset atIndex:1];
        [enc setBuffer:buf_k.buffer  offset:buf_k.offset  atIndex:2];
        [enc setBuffer:out.buffer    offset:out.offset    atIndex:3];
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

    bool on_mps = coords.device().is_mps();
    if (on_mps) {
        hashmap_keys = hashmap_keys.contiguous();
        hashmap_values = hashmap_values.contiguous();
    } else {
        hashmap_keys = hashmap_keys.cpu().contiguous();
        hashmap_values = hashmap_values.cpu().contiguous();
    }
    uint32_t M = (uint32_t)coords.size(0);

    auto buf_coords = from_tensor(coords);
    auto buf_vals = from_tensor(values);

    if (hashmap_keys.dtype() == torch::kUInt64) {
        uint32_t N = (uint32_t)hashmap_keys.size(0);

        if (on_mps) {
            // Direct u64: zero-copy against the u64 tensor via atomic_ulong.
            auto buf_hk = from_tensor_inplace(hashmap_keys);
            auto buf_hv = from_tensor_inplace(hashmap_values);
            ctx().dispatch_mps("hashmap_insert_3d_u64_packed", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:buf_hk.buffer     offset:buf_hk.offset     atIndex:0];
                [enc setBuffer:buf_hv.buffer     offset:buf_hv.offset     atIndex:1];
                [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:2];
                [enc setBuffer:buf_vals.buffer   offset:buf_vals.offset   atIndex:3];
                [enc setBytes:&N length:sizeof(N) atIndex:4];
                [enc setBytes:&M length:sizeof(M) atIndex:5];
                [enc setBytes:&W length:sizeof(W) atIndex:6];
                [enc setBytes:&H length:sizeof(H) atIndex:7];
                [enc setBytes:&D length:sizeof(D) atIndex:8];
            }, M);
            return;
        }

        // CPU path: split into hi/lo u32 pairs via from_blob.
        auto keys_flat = hashmap_keys.view({-1}).contiguous();
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)N * 2}, torch::kUInt32).clone();
        auto keys_lo = keys_u32.slice(0, 0, N * 2, 2).contiguous();  // even indices
        auto keys_hi = keys_u32.slice(0, 1, N * 2, 2).contiguous();  // odd indices

        auto buf_hi = from_tensor_inplace(keys_hi);
        auto buf_lo = from_tensor_inplace(keys_lo);
        auto buf_hv = from_tensor_inplace(hashmap_values);

        ctx().dispatch("hashmap_insert_3d_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_lo.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:3];
            [enc setBuffer:buf_vals.buffer   offset:buf_vals.offset   atIndex:4];
            [enc setBytes:&N length:sizeof(N) atIndex:5];
            [enc setBytes:&M length:sizeof(M) atIndex:6];
            [enc setBytes:&W length:sizeof(W) atIndex:7];
            [enc setBytes:&H length:sizeof(H) atIndex:8];
            [enc setBytes:&D length:sizeof(D) atIndex:9];
        }, M);

        if (hashmap_values.data_ptr() != buf_hv.backing.data_ptr())
            memcpy(hashmap_values.data_ptr(), buf_hv.backing.data_ptr(), hashmap_values.nbytes());
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

        dispatch_auto(on_mps, "hashmap_insert_3d_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer     offset:buf_hk.offset     atIndex:0];
            [enc setBuffer:buf_hv.buffer     offset:buf_hv.offset     atIndex:1];
            [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:2];
            [enc setBuffer:buf_vals.buffer   offset:buf_vals.offset   atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&M length:sizeof(M) atIndex:5];
            [enc setBytes:&W length:sizeof(W) atIndex:6];
            [enc setBytes:&H length:sizeof(H) atIndex:7];
            [enc setBytes:&D length:sizeof(D) atIndex:8];
        }, M);

        if (!on_mps) {
            if (hashmap_keys.data_ptr() != buf_hk.backing.data_ptr())
                memcpy(hashmap_keys.data_ptr(), buf_hk.backing.data_ptr(), hashmap_keys.nbytes());
            if (hashmap_values.data_ptr() != buf_hv.backing.data_ptr())
                memcpy(hashmap_values.data_ptr(), buf_hv.backing.data_ptr(), hashmap_values.nbytes());
        }
    }
}

torch::Tensor hashmap_lookup_3d_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    int W, int H, int D
) {
    bool on_mps = coords.device().is_mps();
    uint32_t N = (uint32_t)hashmap_keys.size(0);
    uint32_t M = (uint32_t)coords.size(0);

    auto buf_coords = from_tensor(coords);
    auto out = make_output({(int64_t)M}, hashmap_values.scalar_type(), coords.device());

    if (hashmap_keys.dtype() == torch::kUInt64 && on_mps) {
        // Direct u64 lookup on MPS.
        auto buf_hk = from_tensor(hashmap_keys);
        auto buf_hv = from_tensor(hashmap_values);
        ctx().dispatch_mps("hashmap_lookup_3d_u64_packed", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer     offset:buf_hk.offset     atIndex:0];
            [enc setBuffer:buf_hv.buffer     offset:buf_hv.offset     atIndex:1];
            [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:2];
            [enc setBuffer:out.buffer        offset:out.offset        atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&M length:sizeof(M) atIndex:5];
            [enc setBytes:&W length:sizeof(W) atIndex:6];
            [enc setBytes:&H length:sizeof(H) atIndex:7];
            [enc setBytes:&D length:sizeof(D) atIndex:8];
        }, M);
        return out.backing;
    }

    if (hashmap_keys.dtype() == torch::kUInt64) {
        // CPU path: split into hi/lo.
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
            [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:3];
            [enc setBuffer:out.buffer        offset:out.offset        atIndex:4];
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

        dispatch_auto(on_mps, "hashmap_lookup_3d_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer     offset:buf_hk.offset     atIndex:0];
            [enc setBuffer:buf_hv.buffer     offset:buf_hv.offset     atIndex:1];
            [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:2];
            [enc setBuffer:out.buffer        offset:out.offset        atIndex:3];
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
    bool on_mps = coords.device().is_mps();
    if (on_mps) {
        hashmap_keys = hashmap_keys.contiguous();
        hashmap_values = hashmap_values.contiguous();
    } else {
        hashmap_keys = hashmap_keys.cpu().contiguous();
        hashmap_values = hashmap_values.cpu().contiguous();
    }
    uint32_t M = (uint32_t)coords.size(0);

    auto buf_coords = from_tensor(coords);

    if (hashmap_keys.dtype() == torch::kUInt64 && on_mps) {
        uint32_t N = (uint32_t)hashmap_keys.size(0);
        auto buf_hk = from_tensor_inplace(hashmap_keys);
        auto buf_hv = from_tensor_inplace(hashmap_values);
        ctx().dispatch_mps("hashmap_insert_3d_idx_as_val_u64_packed", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer     offset:buf_hk.offset     atIndex:0];
            [enc setBuffer:buf_hv.buffer     offset:buf_hv.offset     atIndex:1];
            [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:2];
            [enc setBytes:&N length:sizeof(N) atIndex:3];
            [enc setBytes:&M length:sizeof(M) atIndex:4];
            [enc setBytes:&W length:sizeof(W) atIndex:5];
            [enc setBytes:&H length:sizeof(H) atIndex:6];
            [enc setBytes:&D length:sizeof(D) atIndex:7];
        }, M);
        return;
    }

    if (hashmap_keys.dtype() == torch::kUInt64) {
        uint32_t N = (uint32_t)hashmap_keys.size(0);
        auto keys_flat = hashmap_keys.view({-1}).contiguous();
        auto keys_u32 = torch::from_blob(keys_flat.data_ptr(), {(int64_t)N * 2}, torch::kUInt32).clone();
        auto keys_lo = keys_u32.slice(0, 0, N * 2, 2).contiguous();
        auto keys_hi = keys_u32.slice(0, 1, N * 2, 2).contiguous();

        auto buf_hi = from_tensor_inplace(keys_hi);
        auto buf_lo = from_tensor_inplace(keys_lo);
        auto buf_hv = from_tensor_inplace(hashmap_values);

        // CPU-only path — uint64 MPS support not yet implemented (TORCH_CHECK above).
        ctx().dispatch("hashmap_insert_3d_idx_as_val_u64", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hi.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_lo.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_hv.buffer offset:0 atIndex:2];
            [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:3];
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

        dispatch_auto(on_mps, "hashmap_insert_3d_idx_as_val_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer     offset:buf_hk.offset     atIndex:0];
            [enc setBuffer:buf_hv.buffer     offset:buf_hv.offset     atIndex:1];
            [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:2];
            [enc setBytes:&N length:sizeof(N) atIndex:3];
            [enc setBytes:&M length:sizeof(M) atIndex:4];
            [enc setBytes:&W length:sizeof(W) atIndex:5];
            [enc setBytes:&H length:sizeof(H) atIndex:6];
            [enc setBytes:&D length:sizeof(D) atIndex:7];
        }, M);

        if (!on_mps) {
            if (hashmap_keys.data_ptr() != buf_hk.backing.data_ptr())
                memcpy(hashmap_keys.data_ptr(), buf_hk.backing.data_ptr(), hashmap_keys.nbytes());
            if (hashmap_values.data_ptr() != buf_hv.backing.data_ptr())
                memcpy(hashmap_values.data_ptr(), buf_hv.backing.data_ptr(), hashmap_values.nbytes());
        }
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
    bool on_mps = coords.device().is_mps();
    uint32_t N_val = (uint32_t)coords.size(0);
    uint32_t bl = (uint32_t)bit_length;

    auto buf_coords = from_tensor(coords);
    if (!on_mps) codes = codes.cpu().contiguous();
    else codes = codes.contiguous();
    auto buf_codes = from_tensor_inplace(codes);

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "z_order_encode_3d_u32" : "z_order_encode_3d_u64";

    dispatch_auto(on_mps, kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:0];
        [enc setBuffer:buf_codes.buffer  offset:buf_codes.offset  atIndex:1];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:2];
        [enc setBytes:&bl length:sizeof(bl) atIndex:3];
    }, N_val);

    if (!on_mps) {
        if (codes.data_ptr() != buf_codes.backing.data_ptr())
            memcpy(codes.data_ptr(), buf_codes.backing.data_ptr(), codes.nbytes());
    }
}

torch::Tensor z_order_decode(
    const torch::Tensor& codes,
    const size_t bit_length
) {
    bool on_mps = codes.device().is_mps();
    uint32_t N_val = (uint32_t)codes.size(0);
    uint32_t bl = (uint32_t)bit_length;

    auto buf_codes = from_tensor(codes);
    auto out = make_output({codes.size(0), 4}, torch::kInt32, codes.device());

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "z_order_decode_3d_u32" : "z_order_decode_3d_u64";

    dispatch_auto(on_mps, kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_codes.buffer offset:buf_codes.offset atIndex:0];
        [enc setBuffer:out.buffer       offset:out.offset       atIndex:1];
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
    bool on_mps = coords.device().is_mps();
    uint32_t N_val = (uint32_t)coords.size(0);
    uint32_t bl = (uint32_t)bit_length;

    auto buf_coords = from_tensor(coords);
    if (!on_mps) codes = codes.cpu().contiguous();
    else codes = codes.contiguous();
    auto buf_codes = from_tensor_inplace(codes);

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "hilbert_encode_3d_u32" : "hilbert_encode_3d_u64";

    dispatch_auto(on_mps, kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:0];
        [enc setBuffer:buf_codes.buffer  offset:buf_codes.offset  atIndex:1];
        [enc setBytes:&N_val length:sizeof(N_val) atIndex:2];
        [enc setBytes:&bl length:sizeof(bl) atIndex:3];
    }, N_val);

    if (!on_mps) {
        if (codes.data_ptr() != buf_codes.backing.data_ptr())
            memcpy(codes.data_ptr(), buf_codes.backing.data_ptr(), codes.nbytes());
    }
}

torch::Tensor hilbert_decode(
    const torch::Tensor& codes,
    const size_t bit_length
) {
    bool on_mps = codes.device().is_mps();
    uint32_t N_val = (uint32_t)codes.size(0);
    uint32_t bl = (uint32_t)bit_length;

    auto buf_codes = from_tensor(codes);
    auto out = make_output({codes.size(0), 4}, torch::kInt32, codes.device());

    std::string kernel_name = (codes.dtype() == torch::kInt32) ? "hilbert_decode_3d_u32" : "hilbert_decode_3d_u64";

    dispatch_auto(on_mps, kernel_name, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_codes.buffer offset:buf_codes.offset atIndex:0];
        [enc setBuffer:out.buffer       offset:out.offset       atIndex:1];
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
    bool on_mps = coords.device().is_mps();

    // Insert coords into hashmap first
    hash::hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D);

    uint32_t N = (uint32_t)hashmap_keys.size(0);
    uint32_t B = (uint32_t)grid.size(0);
    uint32_t L = (uint32_t)grid.size(1);

    auto buf_neigh = make_output_filled({(int64_t)B, (int64_t)L}, torch::kUInt32, (int64_t)0xFFFFFFFF, grid.device());

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_vals);
    auto buf_grid = from_tensor(grid);

    if (hashmap_keys.dtype() == torch::kUInt64) {
        // CPU-only path — uint64 MPS support not yet implemented (gated upstream).
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
            [enc setBuffer:buf_grid.buffer  offset:buf_grid.offset  atIndex:3];
            [enc setBuffer:buf_neigh.buffer offset:buf_neigh.offset atIndex:4];
            [enc setBytes:&N length:sizeof(N) atIndex:5];
            [enc setBytes:&B length:sizeof(B) atIndex:6];
            [enc setBytes:&L length:sizeof(L) atIndex:7];
            [enc setBytes:&W length:sizeof(W) atIndex:8];
            [enc setBytes:&H length:sizeof(H) atIndex:9];
            [enc setBytes:&D length:sizeof(D) atIndex:10];
        }, B * L);
    } else {
        dispatch_auto(on_mps, "grid_sample_nearest_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer    offset:buf_hk.offset    atIndex:0];
            [enc setBuffer:buf_hv.buffer    offset:buf_hv.offset    atIndex:1];
            [enc setBuffer:buf_grid.buffer  offset:buf_grid.offset  atIndex:2];
            [enc setBuffer:buf_neigh.buffer offset:buf_neigh.offset atIndex:3];
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
    bool on_mps = coords.device().is_mps();

    hash::hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D);

    uint32_t N = (uint32_t)hashmap_keys.size(0);
    uint32_t B = (uint32_t)grid.size(0);
    uint32_t L = (uint32_t)grid.size(1);

    auto buf_neigh = make_output_filled({(int64_t)B, (int64_t)L, 8}, torch::kUInt32, (int64_t)0xFFFFFFFF, grid.device());
    auto buf_weight = make_output_zeroed({(int64_t)B, (int64_t)L, 8}, torch::kFloat32, grid.device());

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_vals);
    auto buf_grid = from_tensor(grid);

    if (hashmap_keys.dtype() == torch::kUInt64) {
        // CPU-only path — uint64 MPS support not yet implemented (gated upstream).
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
            [enc setBuffer:buf_grid.buffer   offset:buf_grid.offset   atIndex:3];
            [enc setBuffer:buf_neigh.buffer  offset:buf_neigh.offset  atIndex:4];
            [enc setBuffer:buf_weight.buffer offset:buf_weight.offset atIndex:5];
            [enc setBytes:&N length:sizeof(N) atIndex:6];
            [enc setBytes:&B length:sizeof(B) atIndex:7];
            [enc setBytes:&L length:sizeof(L) atIndex:8];
            [enc setBytes:&W length:sizeof(W) atIndex:9];
            [enc setBytes:&H length:sizeof(H) atIndex:10];
            [enc setBytes:&D length:sizeof(D) atIndex:11];
        }, B * L);
    } else {
        dispatch_auto(on_mps, "grid_sample_trilinear_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer     offset:buf_hk.offset     atIndex:0];
            [enc setBuffer:buf_hv.buffer     offset:buf_hv.offset     atIndex:1];
            [enc setBuffer:buf_grid.buffer   offset:buf_grid.offset   atIndex:2];
            [enc setBuffer:buf_neigh.buffer  offset:buf_neigh.offset  atIndex:3];
            [enc setBuffer:buf_weight.buffer offset:buf_weight.offset atIndex:4];
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
    bool on_mps = input.device().is_mps();
    uint32_t M = (uint32_t)indices.size(0);
    uint32_t C = (uint32_t)input.size(1);
    uint32_t V = (uint32_t)weight.size(1);

    auto buf_in = from_tensor(input);
    auto buf_idx = from_tensor(indices);
    auto buf_w = from_tensor(weight);
    auto out = make_output({(int64_t)M, (int64_t)C}, input.scalar_type(), input.device());

    uint32_t tg_x = std::min(C, (uint32_t)32);
    uint32_t tg_y = std::min(M, (uint32_t)(256 / tg_x));
    MTLSize grid = MTLSizeMake((C + tg_x - 1) / tg_x, (M + tg_y - 1) / tg_y, 1);
    MTLSize threadgroup = MTLSizeMake(tg_x, tg_y, 1);

    dispatch_2d_auto(on_mps, "indice_weighted_sum_fwd", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_in.buffer  offset:buf_in.offset  atIndex:0];
        [enc setBuffer:buf_idx.buffer offset:buf_idx.offset atIndex:1];
        [enc setBuffer:buf_w.buffer   offset:buf_w.offset   atIndex:2];
        [enc setBuffer:out.buffer     offset:out.offset     atIndex:3];
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
    bool on_mps = grad_output.device().is_mps();
    uint32_t M = (uint32_t)indices.size(0);
    uint32_t C = (uint32_t)grad_output.size(1);
    uint32_t V = (uint32_t)weight.size(1);

    auto buf_go = from_tensor(grad_output);
    auto buf_idx = from_tensor(indices);
    auto buf_w = from_tensor(weight);
    auto out = make_output_zeroed({N, (int64_t)C}, grad_output.scalar_type(), grad_output.device());

    uint32_t MV = M * V;
    uint32_t tg_x = std::min(C, (uint32_t)32);
    uint32_t tg_y = std::min(MV, (uint32_t)(256 / tg_x));
    MTLSize grid = MTLSizeMake((C + tg_x - 1) / tg_x, (MV + tg_y - 1) / tg_y, 1);
    MTLSize threadgroup = MTLSizeMake(tg_x, tg_y, 1);

    dispatch_2d_auto(on_mps, "indice_weighted_sum_bwd_input", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_go.buffer  offset:buf_go.offset  atIndex:0];
        [enc setBuffer:buf_idx.buffer offset:buf_idx.offset atIndex:1];
        [enc setBuffer:buf_w.buffer   offset:buf_w.offset   atIndex:2];
        [enc setBuffer:out.buffer     offset:out.offset     atIndex:3];
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
    bool on_mps = coords.device().is_mps();
    int V = Kw * Kh * Kd;

    hash::hashmap_insert_3d_idx_as_val_cuda(hashmap_keys, hashmap_vals, coords, W, H, D);

    uint32_t hash_N = (uint32_t)hashmap_keys.size(0);
    uint32_t M = (uint32_t)coords.size(0);
    uint64_t thread_count = (uint64_t)M * (V / 2 + 1);

    auto buf_neigh = make_output_filled({coords.size(0), (int64_t)V}, torch::kUInt32, (int64_t)0xFFFFFFFF, coords.device());

    auto buf_hk = from_tensor(hashmap_keys);
    auto buf_hv = from_tensor(hashmap_vals);
    auto buf_coords = from_tensor(coords);

    if (hashmap_keys.dtype() == torch::kUInt64) {
        if (on_mps) {
            // Direct u64 path — take the u64 tensor zero-copy and run the
            // atomic_ulong kernel (requires Metal 4.0 and Apple Silicon;
            // gated by -D__HAVE_ATOMIC_ULONG__=1 in setup.py).
            ctx().dispatch_mps("submanifold_conv_neighbor_map_u64_packed", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:buf_hk.buffer     offset:buf_hk.offset     atIndex:0];
                [enc setBuffer:buf_hv.buffer     offset:buf_hv.offset     atIndex:1];
                [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:2];
                [enc setBuffer:buf_neigh.buffer  offset:buf_neigh.offset  atIndex:3];
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
        } else {
            // CPU path: decompose u64 into (hi, lo) u32 tensors (via
            // torch::from_blob) so the slice-as-view trick gives us two
            // Metal buffers without a separate allocation.
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
                [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:3];
                [enc setBuffer:buf_neigh.buffer  offset:buf_neigh.offset  atIndex:4];
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
        }
    } else {
        dispatch_auto(on_mps, "submanifold_conv_neighbor_map_u32", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_hk.buffer     offset:buf_hk.offset     atIndex:0];
            [enc setBuffer:buf_hv.buffer     offset:buf_hv.offset     atIndex:1];
            [enc setBuffer:buf_coords.buffer offset:buf_coords.offset atIndex:2];
            [enc setBuffer:buf_neigh.buffer  offset:buf_neigh.offset  atIndex:3];
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
    bool on_mps = neighbor_map.device().is_mps();
    auto device = neighbor_map.device();
    int64_t N = neighbor_map.size(0);
    int64_t V = neighbor_map.size(1);

    uint32_t N32 = (uint32_t)N;
    uint32_t V32 = (uint32_t)V;
    uint32_t tg_size = 256;
    uint32_t num_groups = (N32 + tg_size - 1) / tg_size;
    uint32_t shared_mem = tg_size * V32 * sizeof(uint32_t);

    auto buf_nm = from_tensor(neighbor_map);
    auto buf_gc = make_output({N}, torch::kInt32, device);
    auto buf_bc = make_output({N}, torch::kInt32, device);
    auto buf_nmt = make_output({V * N}, torch::kUInt32, device);
    auto buf_nmaskt = make_output({V * N}, torch::kInt32, device);

    // Dispatch gray_code kernel — orders naturally via MPS stream / batch on CPU.
    if (on_mps) {
        ctx().dispatch_2d_mps("neighbor_map_gray_code", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_nm.buffer      offset:buf_nm.offset      atIndex:0];
            [enc setBuffer:buf_gc.buffer      offset:buf_gc.offset      atIndex:1];
            [enc setBuffer:buf_bc.buffer      offset:buf_bc.offset      atIndex:2];
            [enc setBuffer:buf_nmt.buffer     offset:buf_nmt.offset     atIndex:3];
            [enc setBuffer:buf_nmaskt.buffer  offset:buf_nmaskt.offset  atIndex:4];
            [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
            [enc setBytes:&V32 length:sizeof(V32) atIndex:6];
            [enc setThreadgroupMemoryLength:shared_mem atIndex:0];
        }, MTLSizeMake(num_groups, 1, 1), MTLSizeMake(tg_size, 1, 1));
    } else {
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
    }

    auto sorted_idx = torch::argsort(buf_bc.backing);

    // Prefix sum and gather
    auto prefix_sum = torch::cumsum(buf_nmaskt.backing, 0, torch::kInt32);
    auto total_valid = prefix_sum[-1].item<int32_t>();

    auto buf_ps = from_tensor(prefix_sum);
    auto buf_nmt2 = from_tensor(buf_nmt.backing);
    auto buf_vso = make_output({total_valid}, torch::kUInt32, device);
    auto buf_vsi = make_output({total_valid}, torch::kUInt32, device);
    auto buf_vss = make_output({V + 1}, torch::kUInt32, device);

    if (on_mps) {
        ctx().dispatch_mps("gather_idx_val_seg_from_prefix_sum", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_ps.buffer   offset:buf_ps.offset   atIndex:0];
            [enc setBuffer:buf_nmt2.buffer offset:buf_nmt2.offset atIndex:1];
            [enc setBuffer:buf_vso.buffer  offset:buf_vso.offset  atIndex:2];
            [enc setBuffer:buf_vsi.buffer  offset:buf_vsi.offset  atIndex:3];
            [enc setBuffer:buf_vss.buffer  offset:buf_vss.offset  atIndex:4];
            [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
            [enc setBytes:&V32 length:sizeof(V32) atIndex:6];
        }, N * V);
    } else {
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
    }

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
    bool on_mps = gray_code.device().is_mps();
    auto device = gray_code.device();
    uint32_t N = (uint32_t)gray_code.size(0);
    auto num_blocks = (int64_t)((N + block_size - 1) / block_size);

    int block_dim = block_size;
    uint32_t tg_size = 256;
    uint32_t num_dispatch_groups = ((N + 1) / 2 + tg_size - 1) / tg_size;
    uint32_t shared_mem = tg_size * sizeof(int32_t);

    auto buf_gc = from_tensor(gray_code);
    auto buf_si = from_tensor(sorted_idx);
    auto buf_rc = make_output({num_blocks}, torch::kInt32, device);
    auto buf_sl = make_output({num_blocks + 1}, torch::kInt32, device);

    if (on_mps) {
        ctx().dispatch_2d_mps("reduce_code", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_gc.buffer offset:buf_gc.offset atIndex:0];
            [enc setBuffer:buf_si.buffer offset:buf_si.offset atIndex:1];
            [enc setBuffer:buf_rc.buffer offset:buf_rc.offset atIndex:2];
            [enc setBuffer:buf_sl.buffer offset:buf_sl.offset atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&block_dim length:sizeof(block_dim) atIndex:5];
            [enc setThreadgroupMemoryLength:shared_mem atIndex:0];
        }, MTLSizeMake(num_dispatch_groups, 1, 1), MTLSizeMake(tg_size, 1, 1));
    } else {
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
    }

    auto seglen = torch::cumsum(buf_sl.backing, 0, torch::kInt32);

    auto total_valid = seglen[-1].item<int32_t>();
    uint32_t nb = (uint32_t)num_blocks;

    auto buf_sl2 = from_tensor(seglen);
    auto buf_rc2 = from_tensor(buf_rc.backing);
    auto buf_vki = make_output({total_valid}, torch::kInt32, device);

    if (on_mps) {
        ctx().dispatch_mps("scatter_reduced_code", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_rc2.buffer offset:buf_rc2.offset atIndex:0];
            [enc setBuffer:buf_sl2.buffer offset:buf_sl2.offset atIndex:1];
            [enc setBuffer:buf_vki.buffer offset:buf_vki.offset atIndex:2];
            [enc setBytes:&nb length:sizeof(nb) atIndex:3];
        }, num_blocks);
    } else {
        ctx().begin_batch();
        ctx().dispatch_batched("scatter_reduced_code", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:buf_rc2.buffer offset:0 atIndex:0];
            [enc setBuffer:buf_sl2.buffer offset:0 atIndex:1];
            [enc setBuffer:buf_vki.buffer offset:0 atIndex:2];
            [enc setBytes:&nb length:sizeof(nb) atIndex:3];
        }, num_blocks);
        ctx().end_batch();
    }

    return std::make_tuple(buf_vki.backing, seglen);
    } // @autoreleasepool
}

// ============================================================================
// Spconv implicit GEMM — Metal compute shader dispatch
// ============================================================================

// Pick the kernel-name suffix for the input dtype.
// fp16 / bf16 use specialized half/bfloat kernels; float32 uses the original.
// Mixed dtypes are not supported — caller must align weight/bias to input dtype.
static const char* gemm_dtype_suffix(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return "";
        case torch::kFloat16: return "_half";
        case torch::kBFloat16: return "_bfloat";
        default:
            TORCH_CHECK(false, "Unsupported dtype for spconv implicit GEMM: ",
                        c10::toString(dtype),
                        " (expected float32, float16, or bfloat16)");
    }
}

// Element size for smem tile allocation.
static size_t gemm_elem_bytes(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return 4;
        case torch::kFloat16: return 2;
        case torch::kBFloat16: return 2;
        default:
            TORCH_CHECK(false, "Unsupported dtype for spconv implicit GEMM");
    }
}

// Threadgroup memory bytes for fwd / bwd_input.
//   float32: smem_a + smem_b + smem_nb + skip-empty-V atomic flag (output reuses smem_a buffer).
//   half/bfloat: smem_a + smem_b + smem_nb + skip-empty-V atomic flag + smem_out (separate fp32 scratch).
static uint32_t gemm_smem_fwd_input(torch::ScalarType dtype) {
    size_t es = gemm_elem_bytes(dtype);
    uint32_t base = (uint32_t)(64 * 32 * es + 32 * 64 * es + 64 * sizeof(uint32_t) + sizeof(uint32_t));
    if (dtype == torch::kFloat32) {
        // Layout: 64*32 + 32*64 floats = 16384 bytes for tiles + 64 uints for smem_nb
        // + 1 uint for the skip-empty-V atomic flag = 16644 bytes. Output reuses smem_a.
        return (uint32_t)((64 * 32 + 32 * 64) * sizeof(float)
                          + 64 * sizeof(uint32_t)
                          + sizeof(uint32_t));
    }
    // Half / bfloat: add a B1*B2 float scratch for accumulator store-back.
    return base + (uint32_t)(64 * 64 * sizeof(float));
}

// Threadgroup memory bytes for masked fwd. Layout adds smem_sorted (B1 ints)
// in place of the skip-empty-V atomic flag.
//   float32: smem_a + smem_b + smem_nb + smem_sorted (output reuses smem_a buffer)
//   half/bfloat: smem_a + smem_b + smem_nb + smem_sorted + smem_out (separate fp32 scratch)
static uint32_t gemm_smem_fwd_masked(torch::ScalarType dtype) {
    size_t es = gemm_elem_bytes(dtype);
    uint32_t base = (uint32_t)(64 * 32 * es + 32 * 64 * es
                               + 64 * sizeof(uint32_t)
                               + 64 * sizeof(int32_t));
    if (dtype == torch::kFloat32) {
        return (uint32_t)((64 * 32 + 32 * 64) * sizeof(float)
                          + 64 * sizeof(uint32_t)
                          + 64 * sizeof(int32_t));
    }
    return base + (uint32_t)(64 * 64 * sizeof(float));
}

// Masked bwd-input shares the masked fwd smem layout (smem_a, smem_b, smem_nb,
// smem_sorted; output reuses smem_a region in fp32 or has a separate smem_out
// scratch in half/bfloat).
static uint32_t gemm_smem_bwd_input_masked(torch::ScalarType dtype) {
    return gemm_smem_fwd_masked(dtype);
}

// Masked bwd-weight smem: smem_a + smem_b + smem_si[BK] + smem_so[BK]
// (+ smem_out[B1*B2] in half/bfloat).
static uint32_t gemm_smem_bwd_weight_masked(torch::ScalarType dtype) {
    size_t es = gemm_elem_bytes(dtype);
    if (dtype == torch::kFloat32) {
        return (uint32_t)((64 * 32 + 32 * 64) * sizeof(float)
                          + 32 * sizeof(uint32_t)
                          + 32 * sizeof(uint32_t));
    }
    uint32_t base = (uint32_t)(64 * 32 * es + 32 * 64 * es
                               + 32 * sizeof(uint32_t)
                               + 32 * sizeof(uint32_t));
    return base + (uint32_t)(64 * 64 * sizeof(float));
}

// Wide-tile (B2=128) kernels exist in spconv_gemm.metal but are gated off by
// default — prior measurement showed the wider tile regressed perf at every
// trellis2 shape (e.g. res=32 ch=128 went from 0.50ms to 0.67ms with tile128).
// Apple Silicon's scheduling appears to prefer many narrow tiles over few wide
// ones for this workload, contrary to the typical NVIDIA pattern. Set
// FLEX_GEMM_TILE128=1 to re-enable at dispatch time for re-profiling.
static bool use_tile128_env() {
    static bool enabled = []() {
        const char* env = std::getenv("FLEX_GEMM_TILE128");
        return env != nullptr && std::string(env) == "1";
    }();
    return enabled;
}

static uint32_t gemm_smem_fwd_input_t128(torch::ScalarType dtype) {
    // smem_a[B1*BK] + smem_b[BK*B2_T128] + smem_nb[B1] + 1 atomic + smem_out[B1*HALF_COLS]
    size_t es = gemm_elem_bytes(dtype);
    // fp32 tile128 kernel is not defined — only half/bfloat have tile128.
    uint32_t base = (uint32_t)(64 * 32 * es + 32 * 128 * es
                               + 64 * sizeof(uint32_t) + sizeof(uint32_t));
    return base + (uint32_t)(64 * 64 * sizeof(float));
}

static uint32_t gemm_smem_fwd_masked_t128(torch::ScalarType dtype) {
    size_t es = gemm_elem_bytes(dtype);
    uint32_t base = (uint32_t)(64 * 32 * es + 32 * 128 * es
                               + 64 * sizeof(uint32_t) + 64 * sizeof(int32_t));
    return base + (uint32_t)(64 * 64 * sizeof(float));
}

// Threadgroup memory bytes for bwd_weight (no smem_nb in this kernel).
static uint32_t gemm_smem_bwd_weight(torch::ScalarType dtype) {
    size_t es = gemm_elem_bytes(dtype);
    if (dtype == torch::kFloat32) {
        // Original layout: 64*32 + 32*64 floats = 16384 bytes.
        return (uint32_t)((64 * 32 + 32 * 64) * sizeof(float));
    }
    uint32_t base = (uint32_t)(64 * 32 * es + 32 * 64 * es);
    return base + (uint32_t)(64 * 64 * sizeof(float));
}

// Whether to time GEMM kernels for autotune. Off by default — synchronous waits
// kill MPS-stream overlap. Set FLEX_GEMM_AUTOTUNE=1 to opt in.
static bool gemm_autotune_enabled() {
    static bool enabled = []() {
        const char* env = std::getenv("FLEX_GEMM_AUTOTUNE");
        return env != nullptr && std::string(env) == "1";
    }();
    return enabled;
}

torch::Tensor spconv_fwd_implicit_gemm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& neighbor
) {
    bool on_mps = input.device().is_mps();
    uint32_t N  = (uint32_t)input.size(0);
    uint32_t Ci = (uint32_t)input.size(1);
    uint32_t Co = (uint32_t)weight.size(0);
    uint32_t V  = (uint32_t)weight.size(1);
    // weight shape: [Co, V, Ci]

    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "spconv_fwd_implicit_gemm: weight dtype must match input dtype");
    if (bias.numel() > 0) {
        TORCH_CHECK(bias.scalar_type() == input.scalar_type(),
                    "spconv_fwd_implicit_gemm: bias dtype must match input dtype");
    }

    auto buf_input    = from_tensor(input);
    auto buf_weight   = from_tensor(weight);
    auto buf_neighbor = from_tensor(neighbor);
    auto out = make_output({(int64_t)N, (int64_t)Co}, input.scalar_type(), input.device());

    // Bias: use zero-copy if present, small dummy alloc otherwise.
    TensorBuffer buf_bias_tb;
    id<MTLBuffer> bias_buf;
    NSUInteger bias_offset = 0;
    if (bias.numel() > 0) {
        buf_bias_tb = from_tensor(bias);
        bias_buf = buf_bias_tb.buffer;
        bias_offset = buf_bias_tb.offset;
    } else {
        bias_buf = alloc(4);
    }
    uint32_t has_bias = (bias.numel() > 0) ? 1 : 0;

    // Grid: (cdiv(N, 64), cdiv(Co, 64 or 128)), Threadgroup: 256.
    // Wide-tile (B2=128) is gated by FLEX_GEMM_TILE128=1. Only fp16/bf16 have
    // tile128 kernels; fp32 stays on the 64x64 tile regardless.
    bool tile128 = use_tile128_env()
                   && input.scalar_type() != torch::kFloat32
                   && Co >= 128;
    uint32_t B2_eff = tile128 ? 128 : 64;
    uint32_t grid_x = (N + 63) / 64;
    uint32_t grid_y = (Co + B2_eff - 1) / B2_eff;
    uint32_t shared_mem = tile128
        ? gemm_smem_fwd_input_t128(input.scalar_type())
        : gemm_smem_fwd_input(input.scalar_type());

    std::string kernel_name = tile128
        ? std::string("spconv_fwd_implicit_gemm_t128") + gemm_dtype_suffix(input.scalar_type())
        : std::string("spconv_fwd_implicit_gemm") + gemm_dtype_suffix(input.scalar_type());

    auto setup = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_input.buffer    offset:buf_input.offset    atIndex:0];
        [enc setBuffer:buf_weight.buffer   offset:buf_weight.offset   atIndex:1];
        [enc setBuffer:bias_buf            offset:bias_offset         atIndex:2];
        [enc setBuffer:buf_neighbor.buffer offset:buf_neighbor.offset atIndex:3];
        [enc setBuffer:out.buffer          offset:out.offset          atIndex:4];
        [enc setBytes:&N length:sizeof(N) atIndex:5];
        [enc setBytes:&Co length:sizeof(Co) atIndex:6];
        [enc setBytes:&Ci length:sizeof(Ci) atIndex:7];
        [enc setBytes:&V length:sizeof(V) atIndex:8];
        [enc setBytes:&has_bias length:sizeof(has_bias) atIndex:9];
        [enc setThreadgroupMemoryLength:shared_mem atIndex:0];
    };

    if (on_mps) {
        // Encode into PyTorch's MPS stream so the kernel orders correctly with
        // surrounding MPS ops, with no per-call CPU sync. PyTorch commits the
        // buffer at its next sync point.
        ctx().dispatch_threadgroups_mps(kernel_name, setup,
                                        MTLSizeMake(grid_x, grid_y, 1),
                                        MTLSizeMake(256, 1, 1));
    } else {
        // CPU path: the output tensor's storage must be valid the moment we
        // return — there is no MPS scheduler to defer reads. Always wait.
        // FLEX_GEMM_AUTOTUNE=1 additionally records wall-clock timing.
        bool tune = gemm_autotune_enabled();
        auto t0 = tune ? std::chrono::high_resolution_clock::now()
                       : std::chrono::high_resolution_clock::time_point{};
        ctx().dispatch_threadgroups(kernel_name, setup,
                                    MTLSizeMake(grid_x, grid_y, 1),
                                    MTLSizeMake(256, 1, 1),
                                    /*wait=*/true);
        if (tune) {
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1 - t0).count();
            g_spconv_timing.record(N, Ci, Co, V, us);
        }
    }

    return out.backing;
}

// Masked implicit GEMM forward — uses precomputed sorted_idx + valid_kernel
// + valid_kernel_seg to iterate only the valid V positions per n-block.
torch::Tensor spconv_fwd_masked_implicit_gemm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& neighbor,
    const torch::Tensor& sorted_idx_i64,
    const torch::Tensor& valid_kernel,
    const torch::Tensor& valid_kernel_seg
) {
    bool on_mps = input.device().is_mps();
    uint32_t N  = (uint32_t)input.size(0);
    uint32_t Ci = (uint32_t)input.size(1);
    uint32_t Co = (uint32_t)weight.size(0);
    uint32_t V  = (uint32_t)weight.size(1);

    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "spconv_fwd_masked_implicit_gemm: weight dtype must match input");
    if (bias.numel() > 0) {
        TORCH_CHECK(bias.scalar_type() == input.scalar_type(),
                    "spconv_fwd_masked_implicit_gemm: bias dtype must match input");
    }
    TORCH_CHECK(valid_kernel.scalar_type() == torch::kInt32,
                "valid_kernel must be int32");
    TORCH_CHECK(valid_kernel_seg.scalar_type() == torch::kInt32,
                "valid_kernel_seg must be int32");

    // sorted_idx is the int64 result of torch::argsort. Metal kernels read int32,
    // so cast once here. The values are row indices in [0, N) — fit comfortably.
    auto sorted_idx_i32 = sorted_idx_i64.scalar_type() == torch::kInt32
        ? sorted_idx_i64
        : sorted_idx_i64.to(torch::kInt32);

    auto buf_input    = from_tensor(input);
    auto buf_weight   = from_tensor(weight);
    auto buf_neighbor = from_tensor(neighbor);
    auto buf_sorted   = from_tensor(sorted_idx_i32);
    auto buf_vk       = from_tensor(valid_kernel);
    auto buf_vks      = from_tensor(valid_kernel_seg);
    auto out = make_output({(int64_t)N, (int64_t)Co}, input.scalar_type(), input.device());

    TensorBuffer buf_bias_tb;
    id<MTLBuffer> bias_buf;
    NSUInteger bias_offset = 0;
    if (bias.numel() > 0) {
        buf_bias_tb = from_tensor(bias);
        bias_buf = buf_bias_tb.buffer;
        bias_offset = buf_bias_tb.offset;
    } else {
        bias_buf = alloc(4);
    }
    uint32_t has_bias = (bias.numel() > 0) ? 1 : 0;

    bool tile128 = use_tile128_env()
                   && input.scalar_type() != torch::kFloat32
                   && Co >= 128;
    uint32_t B2_eff = tile128 ? 128 : 64;
    uint32_t grid_x = (N + 63) / 64;
    uint32_t grid_y = (Co + B2_eff - 1) / B2_eff;
    uint32_t shared_mem = tile128
        ? gemm_smem_fwd_masked_t128(input.scalar_type())
        : gemm_smem_fwd_masked(input.scalar_type());

    std::string kernel_name = tile128
        ? std::string("spconv_fwd_masked_implicit_gemm_t128") + gemm_dtype_suffix(input.scalar_type())
        : std::string("spconv_fwd_masked_implicit_gemm") + gemm_dtype_suffix(input.scalar_type());

    auto setup = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_input.buffer    offset:buf_input.offset    atIndex:0];
        [enc setBuffer:buf_weight.buffer   offset:buf_weight.offset   atIndex:1];
        [enc setBuffer:bias_buf            offset:bias_offset         atIndex:2];
        [enc setBuffer:buf_neighbor.buffer offset:buf_neighbor.offset atIndex:3];
        [enc setBuffer:buf_sorted.buffer   offset:buf_sorted.offset   atIndex:4];
        [enc setBuffer:buf_vk.buffer       offset:buf_vk.offset       atIndex:5];
        [enc setBuffer:buf_vks.buffer      offset:buf_vks.offset      atIndex:6];
        [enc setBuffer:out.buffer          offset:out.offset          atIndex:7];
        [enc setBytes:&N        length:sizeof(N)        atIndex:8];
        [enc setBytes:&Co       length:sizeof(Co)       atIndex:9];
        [enc setBytes:&Ci       length:sizeof(Ci)       atIndex:10];
        [enc setBytes:&V        length:sizeof(V)        atIndex:11];
        [enc setBytes:&has_bias length:sizeof(has_bias) atIndex:12];
        [enc setThreadgroupMemoryLength:shared_mem atIndex:0];
    };

    if (on_mps) {
        ctx().dispatch_threadgroups_mps(kernel_name, setup,
                                        MTLSizeMake(grid_x, grid_y, 1),
                                        MTLSizeMake(256, 1, 1));
    } else {
        bool tune = gemm_autotune_enabled();
        auto t0 = tune ? std::chrono::high_resolution_clock::now()
                       : std::chrono::high_resolution_clock::time_point{};
        ctx().dispatch_threadgroups(kernel_name, setup,
                                    MTLSizeMake(grid_x, grid_y, 1),
                                    MTLSizeMake(256, 1, 1),
                                    /*wait=*/true);
        if (tune) {
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1 - t0).count();
            g_spconv_timing.record(N, Ci, Co, V, us);
        }
    }

    return out.backing;
}

std::tuple<torch::Tensor, torch::Tensor> spconv_bwd_implicit_gemm(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& neighbor
) {
    bool on_mps = input.device().is_mps();
    uint32_t N  = (uint32_t)input.size(0);
    uint32_t Ci = (uint32_t)input.size(1);
    uint32_t Co = (uint32_t)weight.size(0);
    uint32_t V  = (uint32_t)weight.size(1);
    uint32_t VCi = V * Ci;

    TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(),
                "spconv_bwd_implicit_gemm: grad_output dtype must match input dtype");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "spconv_bwd_implicit_gemm: weight dtype must match input dtype");

    auto buf_go       = from_tensor(grad_output);
    auto buf_input    = from_tensor(input);
    auto buf_weight   = from_tensor(weight);
    auto buf_neighbor = from_tensor(neighbor);
    auto out_gi       = make_output({(int64_t)N, (int64_t)Ci}, input.scalar_type(), input.device());
    auto out_gw       = make_output({(int64_t)Co, (int64_t)V, (int64_t)Ci}, weight.scalar_type(), input.device());

    uint32_t shared_mem_input  = gemm_smem_fwd_input(input.scalar_type());
    uint32_t shared_mem_weight = gemm_smem_bwd_weight(input.scalar_type());

    const char* suffix = gemm_dtype_suffix(input.scalar_type());
    std::string kernel_input  = std::string("spconv_bwd_input_implicit_gemm")  + suffix;
    std::string kernel_weight = std::string("spconv_bwd_weight_implicit_gemm") + suffix;

    auto setup_gi = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_go.buffer       offset:buf_go.offset       atIndex:0];
        [enc setBuffer:buf_weight.buffer   offset:buf_weight.offset   atIndex:1];
        [enc setBuffer:buf_neighbor.buffer offset:buf_neighbor.offset atIndex:2];
        [enc setBuffer:out_gi.buffer       offset:out_gi.offset       atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&Co length:sizeof(Co) atIndex:5];
        [enc setBytes:&Ci length:sizeof(Ci) atIndex:6];
        [enc setBytes:&V length:sizeof(V) atIndex:7];
        [enc setThreadgroupMemoryLength:shared_mem_input atIndex:0];
    };

    auto setup_gw = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_go.buffer       offset:buf_go.offset       atIndex:0];
        [enc setBuffer:buf_input.buffer    offset:buf_input.offset    atIndex:1];
        [enc setBuffer:buf_neighbor.buffer offset:buf_neighbor.offset atIndex:2];
        [enc setBuffer:out_gw.buffer       offset:out_gw.offset       atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&Co length:sizeof(Co) atIndex:5];
        [enc setBytes:&Ci length:sizeof(Ci) atIndex:6];
        [enc setBytes:&V length:sizeof(V) atIndex:7];
        [enc setThreadgroupMemoryLength:shared_mem_weight atIndex:0];
    };

    uint32_t grid_x_gi = (N + 63) / 64;
    uint32_t grid_y_gi = (Ci + 63) / 64;
    uint32_t grid_x_gw = (Co + 63) / 64;
    uint32_t grid_y_gw = (VCi + 63) / 64;

    if (on_mps) {
        // MPS stream serializes the two kernels in command-buffer order; no
        // explicit per-kernel wait needed. PyTorch commits when ready.
        ctx().dispatch_threadgroups_mps(kernel_input,  setup_gi,
                                        MTLSizeMake(grid_x_gi, grid_y_gi, 1),
                                        MTLSizeMake(256, 1, 1));
        ctx().dispatch_threadgroups_mps(kernel_weight, setup_gw,
                                        MTLSizeMake(grid_x_gw, grid_y_gw, 1),
                                        MTLSizeMake(256, 1, 1));
    } else {
        // CPU outputs need synchronous completion before return. Wait once,
        // after the second kernel — the queue serializes them via FIFO.
        ctx().dispatch_threadgroups(kernel_input,  setup_gi,
                                    MTLSizeMake(grid_x_gi, grid_y_gi, 1),
                                    MTLSizeMake(256, 1, 1),
                                    /*wait=*/false);
        ctx().dispatch_threadgroups(kernel_weight, setup_gw,
                                    MTLSizeMake(grid_x_gw, grid_y_gw, 1),
                                    MTLSizeMake(256, 1, 1),
                                    /*wait=*/true);
    }

    return std::make_tuple(out_gi.backing, out_gw.backing);
}

// Masked implicit GEMM backward — uses precomputed sorted_idx + valid_kernel
// + valid_kernel_seg for grad_input (identical V-set as fwd under u=V-1-v),
// and valid_signal_i + valid_signal_o + valid_signal_seg for grad_weight.
std::tuple<torch::Tensor, torch::Tensor> spconv_bwd_masked_implicit_gemm(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& neighbor,
    const torch::Tensor& sorted_idx_i64,
    const torch::Tensor& valid_kernel,
    const torch::Tensor& valid_kernel_seg,
    const torch::Tensor& valid_signal_i,
    const torch::Tensor& valid_signal_o,
    const torch::Tensor& valid_signal_seg
) {
    bool on_mps = input.device().is_mps();
    uint32_t N  = (uint32_t)input.size(0);
    uint32_t Ci = (uint32_t)input.size(1);
    uint32_t Co = (uint32_t)weight.size(0);
    uint32_t V  = (uint32_t)weight.size(1);

    TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(),
                "spconv_bwd_masked_implicit_gemm: grad_output dtype must match input");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "spconv_bwd_masked_implicit_gemm: weight dtype must match input");
    TORCH_CHECK(valid_kernel.scalar_type() == torch::kInt32,
                "valid_kernel must be int32");
    TORCH_CHECK(valid_kernel_seg.scalar_type() == torch::kInt32,
                "valid_kernel_seg must be int32");
    TORCH_CHECK(valid_signal_i.scalar_type() == torch::kUInt32,
                "valid_signal_i must be uint32");
    TORCH_CHECK(valid_signal_o.scalar_type() == torch::kUInt32,
                "valid_signal_o must be uint32");
    TORCH_CHECK(valid_signal_seg.scalar_type() == torch::kUInt32,
                "valid_signal_seg must be uint32");

    // sorted_idx from torch::argsort is int64; Metal kernel reads int32.
    auto sorted_idx_i32 = sorted_idx_i64.scalar_type() == torch::kInt32
        ? sorted_idx_i64
        : sorted_idx_i64.to(torch::kInt32);

    auto buf_go       = from_tensor(grad_output);
    auto buf_input    = from_tensor(input);
    auto buf_weight   = from_tensor(weight);
    auto buf_neighbor = from_tensor(neighbor);
    auto buf_sorted   = from_tensor(sorted_idx_i32);
    auto buf_vk       = from_tensor(valid_kernel);
    auto buf_vks      = from_tensor(valid_kernel_seg);
    auto buf_vsi      = from_tensor(valid_signal_i);
    auto buf_vso      = from_tensor(valid_signal_o);
    auto buf_vss      = from_tensor(valid_signal_seg);
    auto out_gi       = make_output({(int64_t)N, (int64_t)Ci}, input.scalar_type(), input.device());
    auto out_gw       = make_output({(int64_t)Co, (int64_t)V, (int64_t)Ci}, weight.scalar_type(), input.device());

    uint32_t shared_mem_input  = gemm_smem_bwd_input_masked(input.scalar_type());
    uint32_t shared_mem_weight = gemm_smem_bwd_weight_masked(input.scalar_type());

    const char* suffix = gemm_dtype_suffix(input.scalar_type());
    std::string kernel_input  = std::string("spconv_bwd_input_masked_implicit_gemm")  + suffix;
    std::string kernel_weight = std::string("spconv_bwd_weight_masked_implicit_gemm") + suffix;

    auto setup_gi = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_go.buffer       offset:buf_go.offset       atIndex:0];
        [enc setBuffer:buf_weight.buffer   offset:buf_weight.offset   atIndex:1];
        [enc setBuffer:buf_neighbor.buffer offset:buf_neighbor.offset atIndex:2];
        [enc setBuffer:buf_sorted.buffer   offset:buf_sorted.offset   atIndex:3];
        [enc setBuffer:buf_vk.buffer       offset:buf_vk.offset       atIndex:4];
        [enc setBuffer:buf_vks.buffer      offset:buf_vks.offset      atIndex:5];
        [enc setBuffer:out_gi.buffer       offset:out_gi.offset       atIndex:6];
        [enc setBytes:&N  length:sizeof(N)  atIndex:7];
        [enc setBytes:&Co length:sizeof(Co) atIndex:8];
        [enc setBytes:&Ci length:sizeof(Ci) atIndex:9];
        [enc setBytes:&V  length:sizeof(V)  atIndex:10];
        [enc setThreadgroupMemoryLength:shared_mem_input atIndex:0];
    };

    auto setup_gw = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_go.buffer    offset:buf_go.offset    atIndex:0];
        [enc setBuffer:buf_input.buffer offset:buf_input.offset atIndex:1];
        [enc setBuffer:buf_vsi.buffer   offset:buf_vsi.offset   atIndex:2];
        [enc setBuffer:buf_vso.buffer   offset:buf_vso.offset   atIndex:3];
        [enc setBuffer:buf_vss.buffer   offset:buf_vss.offset   atIndex:4];
        [enc setBuffer:out_gw.buffer    offset:out_gw.offset    atIndex:5];
        [enc setBytes:&N  length:sizeof(N)  atIndex:6];
        [enc setBytes:&Co length:sizeof(Co) atIndex:7];
        [enc setBytes:&Ci length:sizeof(Ci) atIndex:8];
        [enc setBytes:&V  length:sizeof(V)  atIndex:9];
        [enc setThreadgroupMemoryLength:shared_mem_weight atIndex:0];
    };

    uint32_t grid_x_gi = (N + 63) / 64;
    uint32_t grid_y_gi = (Ci + 63) / 64;
    uint32_t grid_x_gw = (Co + 63) / 64;
    uint32_t grid_y_gw = (Ci + 63) / 64;

    if (on_mps) {
        ctx().dispatch_threadgroups_mps(kernel_input,  setup_gi,
                                        MTLSizeMake(grid_x_gi, grid_y_gi, 1),
                                        MTLSizeMake(256, 1, 1));
        ctx().dispatch_threadgroups_mps(kernel_weight, setup_gw,
                                        MTLSizeMake(grid_x_gw, grid_y_gw, V),
                                        MTLSizeMake(256, 1, 1));
    } else {
        ctx().dispatch_threadgroups(kernel_input,  setup_gi,
                                    MTLSizeMake(grid_x_gi, grid_y_gi, 1),
                                    MTLSizeMake(256, 1, 1),
                                    /*wait=*/false);
        ctx().dispatch_threadgroups(kernel_weight, setup_gw,
                                    MTLSizeMake(grid_x_gw, grid_y_gw, V),
                                    MTLSizeMake(256, 1, 1),
                                    /*wait=*/true);
    }

    return std::make_tuple(out_gi.backing, out_gw.backing);
}

} // namespace spconv

// ============================================================================
// Fused variable-length sparse attention forward
// ============================================================================
namespace sparse_attn {

torch::Tensor sparse_attention_fwd(
    const torch::Tensor& q,              // [T_q, H, C_q]
    const torch::Tensor& k,              // [T_kv, H, C_q]
    const torch::Tensor& v,              // [T_kv, H, C_v]
    const torch::Tensor& cu_seqlens_q,   // [N+1]
    const torch::Tensor& cu_seqlens_kv,  // [N+1]
    int64_t max_q_seqlen,
    int64_t max_kv_seqlen,
    double scale
) {
    (void)max_kv_seqlen;  // informational; not currently used in the kernel

    bool on_mps = q.device().is_mps();

    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "sparse_attention_fwd: q/k/v must be 3-D [T, H, C]");
    TORCH_CHECK(k.size(0) == v.size(0), "T_kv mismatch between k and v");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
                "H mismatch across q/k/v");
    TORCH_CHECK(q.size(2) == k.size(2),
                "C_q mismatch between q and k");

    int64_t T_q  = q.size(0);
    int64_t H    = q.size(1);
    int64_t C_q  = q.size(2);
    int64_t C_v  = v.size(2);
    TORCH_CHECK(C_q <= 128 && C_v <= 128,
                "sparse_attention_fwd: head dim must be <= 128 (C_q=", C_q,
                ", C_v=", C_v, ")");

    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "q/k/v dtype must all match");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 &&
                cu_seqlens_kv.scalar_type() == torch::kInt32,
                "cu_seqlens_* must be int32");
    TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_kv.size(0),
                "cu_seqlens_q and cu_seqlens_kv must have the same length");

    int64_t N = cu_seqlens_q.size(0) - 1;

    auto buf_q   = from_tensor(q);
    auto buf_k   = from_tensor(k);
    auto buf_v   = from_tensor(v);
    auto buf_csq = from_tensor(cu_seqlens_q);
    auto buf_cskv = from_tensor(cu_seqlens_kv);
    auto out = make_output({T_q, H, C_v}, q.scalar_type(), q.device());

    uint32_t H_u = (uint32_t)H;
    uint32_t Cq_u = (uint32_t)C_q;
    uint32_t Cv_u = (uint32_t)C_v;
    float scale_f = (float)scale;

    const char* suffix = spconv::gemm_dtype_suffix(q.scalar_type());
    std::string kernel_name = std::string("sparse_attention_fwd") + suffix;

    auto setup = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buf_q.buffer    offset:buf_q.offset    atIndex:0];
        [enc setBuffer:buf_k.buffer    offset:buf_k.offset    atIndex:1];
        [enc setBuffer:buf_v.buffer    offset:buf_v.offset    atIndex:2];
        [enc setBuffer:buf_csq.buffer  offset:buf_csq.offset  atIndex:3];
        [enc setBuffer:buf_cskv.buffer offset:buf_cskv.offset atIndex:4];
        [enc setBuffer:out.buffer      offset:out.offset      atIndex:5];
        [enc setBytes:&H_u     length:sizeof(H_u)     atIndex:6];
        [enc setBytes:&Cq_u    length:sizeof(Cq_u)    atIndex:7];
        [enc setBytes:&Cv_u    length:sizeof(Cv_u)    atIndex:8];
        [enc setBytes:&scale_f length:sizeof(scale_f) atIndex:9];
    };

    // Grid: (max_q_seqlen, H, N). Threadgroup: 32 threads in x dimension, 1
    // head per tg. Threads beyond the per-sequence q_len early-exit inside
    // the kernel.
    NSUInteger tg_x = std::min((NSUInteger)32, (NSUInteger)std::max<int64_t>(max_q_seqlen, 1));
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    MTLSize grid_tgs = MTLSizeMake(
        ((NSUInteger)max_q_seqlen + tg_x - 1) / tg_x,
        (NSUInteger)H,
        (NSUInteger)N);

    if (on_mps) {
        ctx().dispatch_threadgroups_mps(kernel_name, setup, grid_tgs, tg);
    } else {
        ctx().dispatch_threadgroups(kernel_name, setup, grid_tgs, tg,
                                    /*wait=*/true);
    }

    return out.backing;
}

} // namespace sparse_attn

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
    m.def("spconv_fwd_masked_implicit_gemm", &spconv::spconv_fwd_masked_implicit_gemm);
    m.def("spconv_bwd_implicit_gemm", &spconv::spconv_bwd_implicit_gemm);
    m.def("spconv_bwd_masked_implicit_gemm", &spconv::spconv_bwd_masked_implicit_gemm);

    // Sparse attention (1)
    m.def("sparse_attention_fwd", &sparse_attn::sparse_attention_fwd);

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

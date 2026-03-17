#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string>
#include <unordered_map>
#include <functional>

namespace flex_gemm {
namespace metal {

class MetalContext {
public:
    static MetalContext& instance();

    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }

    id<MTLComputePipelineState> pipeline(const std::string& kernel_name);

    // Synchronous 1D dispatch
    void dispatch(
        const std::string& kernel_name,
        std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
        uint64_t thread_count
    );

    // Synchronous 2D dispatch with explicit grid/threadgroup
    void dispatch_2d(
        const std::string& kernel_name,
        std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
        MTLSize grid_size,
        MTLSize threadgroup_size
    );

    void synchronize();

private:
    MetalContext();
    ~MetalContext() = default;
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache_;
};

} // namespace metal
} // namespace flex_gemm

#import "metal_context.h"
#include <stdexcept>
#include <dlfcn.h>
#include "dtypes.h"

namespace flex_gemm {
namespace metal {

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::MetalContext() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("No Metal device found");
    }

    queue_ = [device_ newCommandQueue];
    if (!queue_) {
        throw std::runtime_error("Failed to create Metal command queue");
    }

    // Load the precompiled metallib from the same directory as the .so
    // The metallib is installed alongside the extension module by setup.py
    NSBundle* bundle = [NSBundle mainBundle];
    NSString* libPath = nil;

    // Try loading from the Python package directory (pip install -e .)
    // The metallib path is set at build time via an environment variable,
    // but at runtime we search relative to this compiled .so
    @autoreleasepool {
        // Get the path of this shared library
        Dl_info dl_info;
        if (dladdr((void*)&MetalContext::instance, &dl_info)) {
            NSString* soPath = [NSString stringWithUTF8String:dl_info.dli_fname];
            NSString* soDir = [soPath stringByDeletingLastPathComponent];
            libPath = [soDir stringByAppendingPathComponent:@"flex_gemm.metallib"];
        }
    }

    if (libPath) {
        NSURL* libURL = [NSURL fileURLWithPath:libPath];
        NSError* error = nil;
        library_ = [device_ newLibraryWithURL:libURL error:&error];
        if (!library_) {
            throw std::runtime_error(
                std::string("Failed to load metallib from ") +
                [libPath UTF8String] + ": " +
                [[error localizedDescription] UTF8String]
            );
        }
    } else {
        throw std::runtime_error("Could not determine metallib path");
    }
}

id<MTLComputePipelineState> MetalContext::pipeline(const std::string& kernel_name) {
    auto it = pipeline_cache_.find(kernel_name);
    if (it != pipeline_cache_.end()) {
        return it->second;
    }

    NSString* name = [NSString stringWithUTF8String:kernel_name.c_str()];
    id<MTLFunction> func = [library_ newFunctionWithName:name];
    if (!func) {
        throw std::runtime_error("Metal kernel not found: " + kernel_name);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:func error:&error];
    if (!pso) {
        throw std::runtime_error(
            "Failed to create pipeline for " + kernel_name + ": " +
            [[error localizedDescription] UTF8String]
        );
    }

    pipeline_cache_[kernel_name] = pso;
    return pso;
}

void MetalContext::dispatch(
    const std::string& kernel_name,
    std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
    uint64_t thread_count
) {
    if (thread_count == 0) return;

    auto pso = pipeline(kernel_name);
    uint64_t threadgroup_size = std::min((uint64_t)pso.maxTotalThreadsPerThreadgroup, (uint64_t)BLOCK_SIZE);
    // Clamp to BLOCK_SIZE
    if (threadgroup_size > 256) threadgroup_size = 256;

    uint64_t grid_size = ((thread_count + threadgroup_size - 1) / threadgroup_size) * threadgroup_size;

    @autoreleasepool {
        id<MTLCommandBuffer> cmdbuf = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdbuf computeCommandEncoder];

        [encoder setComputePipelineState:pso];
        buffer_setup(encoder);

        [encoder dispatchThreads:MTLSizeMake(grid_size, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
        [encoder endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }
}

void MetalContext::dispatch_2d(
    const std::string& kernel_name,
    std::function<void(id<MTLComputeCommandEncoder>)> buffer_setup,
    MTLSize grid_size,
    MTLSize threadgroup_size
) {
    auto pso = pipeline(kernel_name);

    @autoreleasepool {
        id<MTLCommandBuffer> cmdbuf = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdbuf computeCommandEncoder];

        [encoder setComputePipelineState:pso];
        buffer_setup(encoder);

        [encoder dispatchThreadgroups:grid_size
                threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }
}

void MetalContext::synchronize() {
    @autoreleasepool {
        id<MTLCommandBuffer> cmdbuf = [queue_ commandBuffer];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }
}

} // namespace metal
} // namespace flex_gemm

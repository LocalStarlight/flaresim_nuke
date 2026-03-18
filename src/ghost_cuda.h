// ============================================================================
// ghost_cuda.h — CUDA ghost rendering launcher declaration
// ============================================================================
#pragma once

#include "ghost.h"
#include "lens.h"

#include <cstddef>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// GpuBufferCache — persistent device-memory allocations across frames.
//
// The caller (typically the FlareSim Iop node instance) owns one of these.
// Pass it to launch_ghost_cuda() every call.  The launcher grows each buffer
// only when the new data is larger than the current allocation, avoiding a
// cudaMalloc + cudaFree pair on every frame.
//
// Internal GPU POD types (GPUPair, GPUSource, etc.) are not exposed in this
// header; the buffers are stored as void* and cast inside ghost_cuda.cu.
// ---------------------------------------------------------------------------

struct GpuBufferCache
{
    // Input parameter buffers (void* hides internal GPU POD types).
    void*  d_surfs = nullptr;  std::size_t surfs_bytes = 0;
    void*  d_pairs = nullptr;  std::size_t pairs_bytes = 0;
    void*  d_src   = nullptr;  std::size_t src_bytes   = 0;
    void*  d_grid  = nullptr;  std::size_t grid_bytes  = 0;
    void*  d_spec  = nullptr;  std::size_t spec_bytes  = 0;

    // Output accumulation buffers — float* so their size is self-documenting.
    // All three channels are always grown together (same pixel count).
    float* d_out_r    = nullptr;
    float* d_out_g    = nullptr;
    float* d_out_b    = nullptr;
    std::size_t out_floats = 0;   // capacity of each channel in floats

    // Free all device allocations and reset capacities to zero.
    // Safe to call multiple times and on a partially-constructed cache.
    void release();

    ~GpuBufferCache() { release(); }

    GpuBufferCache()  = default;
    GpuBufferCache(const GpuBufferCache&)            = delete;
    GpuBufferCache& operator=(const GpuBufferCache&) = delete;
};

// One wavelength sample with pre-computed RGB colour weights.
// rw/gw/bw are normalised so the sum across all samples for each channel
// equals 1.0 (energy-preserving across different spectral_samples counts).
struct GPUSpectralSample
{
    float lambda; // wavelength in nm
    float rw, gw, bw; // linear RGB contribution weights
};

// Launch the CUDA ghost rendering kernel.
//
// active_pairs / pair_area_boosts must already be pre-filtered (below-threshold
// pairs removed, area-normalisation boosts computed) — see filter_ghost_pairs()
// in ghost.cpp.
//
// sensor_half_w/h: half-dimensions of the sensor in mm, computed from focal
// length and field of view (used for pixel-to-sensor coordinate mapping).
//
// out_r/g/b: CPU-side output buffers, width×height (bounding-box sized), must
// be zeroed by caller.  After return they contain the accumulated ghost
// contribution in linear light.
//
// fmt_w/h: pixel dimensions of the output format (may be smaller than the
// bounding box when off-frame content is present).
// fmt_x0_in_buf / fmt_y0_in_buf: pixel offset of the format origin within
// the output buffer, i.e. (format.x() - bbox_x0) and (format.y() - bbox_y0).
// Ghost positions are always computed relative to the format centre so that
// the optical axis stays fixed even when upstream nodes push the bounding box
// beyond the frame edges.
//
// out_error: if non-null and a CUDA error occurs, receives an error message.
void launch_ghost_cuda(
    const LensSystem&               lens,
    const std::vector<GhostPair>&   active_pairs,
    const std::vector<float>&       pair_area_boosts,
    const std::vector<BrightPixel>& sources,
    float                           sensor_half_w,
    float                           sensor_half_h,
    float*                          out_r,
    float*                          out_g,
    float*                          out_b,
    int                             width,
    int                             height,
    int                             fmt_w,
    int                             fmt_h,
    int                             fmt_x0_in_buf,
    int                             fmt_y0_in_buf,
    const GhostConfig&              config,
    GpuBufferCache&                 cache,
    std::string*                    out_error = nullptr
);

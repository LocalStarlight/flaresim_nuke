// ============================================================================
// ghost_cuda.cu — GPU ghost rendering: device math, scatter kernel, launcher
//
// Design notes
// ─────────────
// • The device-side math is self-contained (no STL) to avoid portability
//   issues with std::clamp / std::sqrt in device code across toolchain
//   versions.  It mirrors the logic in fresnel.h, vec3.h and trace.cpp
//   exactly — keep them in sync if those files change.
//
// • Surface (from lens.h) is plain POD and is uploaded to device as-is via
//   cudaMemcpy.  Only the flat array is uploaded; LensSystem (which owns a
//   std::vector) stays on the CPU.
//
// • Thread layout:
//     gridDim  = (n_pairs * n_sources, grid_blocks, 1)
//     blockDim = (BLOCK_SIZE, 1, 1)
//   Each thread handles one entrance-pupil sample for one (pair, source) and
//   traces 3 wavelengths (R, G, B) sequentially.  AtomicAdd scatters results.
// ============================================================================

#include "ghost_cuda.h"
#include "lens.h"       // Surface (POD) — uploaded to device
#include "ghost.h"      // BrightPixel, GhostPair, GhostConfig (host side)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>


// ===========================================================================
// GpuBufferCache — implementation of the persistent buffer cache.
// ===========================================================================

void GpuBufferCache::release()
{
    cudaFree(d_surfs);  d_surfs = nullptr;  surfs_bytes = 0;
    cudaFree(d_pairs);  d_pairs = nullptr;  pairs_bytes = 0;
    cudaFree(d_src);    d_src   = nullptr;  src_bytes   = 0;
    cudaFree(d_grid);   d_grid  = nullptr;  grid_bytes  = 0;
    cudaFree(d_spec);   d_spec  = nullptr;  spec_bytes  = 0;
    cudaFree(d_out_r);  d_out_r = nullptr;
    cudaFree(d_out_g);  d_out_g = nullptr;
    cudaFree(d_out_b);  d_out_b = nullptr;
    out_floats = 0;
}

// ===========================================================================
// Device-side math — no STL, no std:: calls
// ===========================================================================

// ---- Minimal 3-vector ----

struct DVec3 { float x, y, z; };

__device__ __forceinline__ DVec3 dv(float x, float y, float z)
    { return {x, y, z}; }
__device__ __forceinline__ DVec3 dv_add(DVec3 a, DVec3 b)
    { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__device__ __forceinline__ DVec3 dv_sub(DVec3 a, DVec3 b)
    { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ __forceinline__ DVec3 dv_scale(DVec3 a, float s)
    { return {a.x*s, a.y*s, a.z*s}; }
__device__ __forceinline__ DVec3 dv_neg(DVec3 a)
    { return {-a.x, -a.y, -a.z}; }
__device__ __forceinline__ float dv_dot(DVec3 a, DVec3 b)
    { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __forceinline__ DVec3 dv_normalize(DVec3 v)
{
    float inv = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return {v.x*inv, v.y*inv, v.z*inv};
}

struct DRay { DVec3 origin; DVec3 dir; };

// ---- Dispersion (Cauchy via Abbe number) ----

__device__ __forceinline__
float d_dispersion_ior(float n_d, float V_d, float lambda_nm)
{
    if (V_d < 0.1f || n_d <= 1.0001f) return n_d;

    const float lF = 486.13f, lC = 656.27f, ld = 587.56f;
    float dn       = (n_d - 1.0f) / V_d;
    float inv_lF2  = 1.0f / (lF * lF);
    float inv_lC2  = 1.0f / (lC * lC);
    float inv_ld2  = 1.0f / (ld * ld);
    float B        = dn / (inv_lF2 - inv_lC2);
    float A        = n_d - B * inv_ld2;
    return A + B / (lambda_nm * lambda_nm);
}

__device__ __forceinline__
float d_ior_at(const Surface& s, float lambda_nm)
{
    return d_dispersion_ior(s.ior, s.abbe_v, lambda_nm);
}

__device__ __forceinline__
float d_ior_before(const Surface* surfs, int idx, float lambda_nm)
{
    return (idx <= 0) ? 1.0f : d_ior_at(surfs[idx - 1], lambda_nm);
}

// ---- Fresnel reflectance ----

__device__ __forceinline__
float d_fresnel_reflectance(float cos_i, float n1, float n2)
{
    cos_i        = fabsf(cos_i);
    float eta    = n1 / n2;
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t >= 1.0f) return 1.0f;  // TIR
    float cos_t  = sqrtf(1.0f - sin2_t);
    float rs     = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t);
    float rp     = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t);
    return 0.5f * (rs*rs + rp*rp);
}

// Single-layer MgF2 AR coating (mirrors coating_reflectance in fresnel.h)
__device__ __forceinline__
float d_coating_reflectance(float cos_i, float n1, float n2,
                             float coating_n, float d_nm, float lambda_nm)
{
    float sin2_c = (n1/coating_n) * (n1/coating_n) * (1.0f - cos_i*cos_i);
    if (sin2_c >= 1.0f) return d_fresnel_reflectance(cos_i, n1, n2);
    float cos_c  = sqrtf(1.0f - sin2_c);

    const float PI = 3.14159265358979323846f;
    float delta  = 2.0f * PI * coating_n * d_nm * cos_c / lambda_nm;
    float r01    = (n1*cos_i - coating_n*cos_c) / (n1*cos_i + coating_n*cos_c);

    float sin2_2 = (coating_n/n2) * (coating_n/n2) * (1.0f - cos_c*cos_c);
    if (sin2_2 >= 1.0f) return d_fresnel_reflectance(cos_i, n1, n2);
    float cos_2  = sqrtf(1.0f - sin2_2);
    float r12    = (coating_n*cos_c - n2*cos_2) / (coating_n*cos_c + n2*cos_2);

    float cos_2d = cosf(2.0f * delta);
    float num    = r01*r01 + r12*r12 + 2.0f*r01*r12*cos_2d;
    float den    = 1.0f   + r01*r01*r12*r12 + 2.0f*r01*r12*cos_2d;
    float R      = num / den;
    return fminf(fmaxf(R, 0.0f), 1.0f);
}

__device__ __forceinline__
float d_surface_reflectance(float cos_i, float n1, float n2,
                             int coating_layers, float lambda_nm)
{
    if (coating_layers <= 0)
        return d_fresnel_reflectance(cos_i, n1, n2);

    // MgF2 quarter-wave at 550 nm
    const float mgf2_n        = 1.38f;
    const float design_lambda = 550.0f;
    float qw_thick            = design_lambda / (4.0f * mgf2_n);  // ~99.6 nm
    float R = d_coating_reflectance(cos_i, n1, n2, mgf2_n, qw_thick, lambda_nm);
    for (int i = 1; i < coating_layers; ++i) R *= 0.25f;
    return fminf(fmaxf(R, 0.0f), 1.0f);
}

// ---- Ray–surface intersection ----

__device__ __forceinline__
bool d_intersect_surface(const DRay& ray, const Surface& surf,
                          DVec3& hit, DVec3& norm)
{
    if (fabsf(surf.radius) < 1e-6f)
    {
        // Flat surface: ray–plane at z = surf.z
        if (fabsf(ray.dir.z) < 1e-12f) return false;
        float t = (surf.z - ray.origin.z) / ray.dir.z;
        if (!(t > 1e-6f)) return false;  // catches t<=0, NaN, and Inf

        hit = dv(ray.origin.x + ray.dir.x*t,
                 ray.origin.y + ray.dir.y*t,
                 ray.origin.z + ray.dir.z*t);

        if (!(hit.x*hit.x + hit.y*hit.y <= surf.semi_aperture*surf.semi_aperture))
            return false;

        norm = dv(0.0f, 0.0f, (ray.dir.z > 0) ? -1.0f : 1.0f);
        return true;
    }

    // Spherical surface — sphere centre at (0, 0, surf.z + surf.radius)
    float R   = surf.radius;
    DVec3 ctr = dv(0.0f, 0.0f, surf.z + R);
    DVec3 oc  = dv_sub(ray.origin, ctr);

    float a    = dv_dot(ray.dir, ray.dir);
    float b    = 2.0f * dv_dot(oc, ray.dir);
    float c    = dv_dot(oc, oc) - R*R;
    float disc = b*b - 4.0f*a*c;
    if (disc < 0.0f) return false;

    float sd    = sqrtf(disc);
    float inv2a = 0.5f / a;
    float t1    = (-b - sd) * inv2a;
    float t2    = (-b + sd) * inv2a;

    float t;
    if (t1 > 1e-6f && t2 > 1e-6f)
    {
        float z1 = ray.origin.z + t1*ray.dir.z;
        float z2 = ray.origin.z + t2*ray.dir.z;
        t = (fabsf(z1 - surf.z) < fabsf(z2 - surf.z)) ? t1 : t2;
    }
    else if (t1 > 1e-6f) t = t1;
    else if (t2 > 1e-6f) t = t2;
    else return false;

    hit = dv(ray.origin.x + ray.dir.x*t,
             ray.origin.y + ray.dir.y*t,
             ray.origin.z + ray.dir.z*t);

    // !(... <= ...) instead of (... >) catches NaN hit points.
    if (!(hit.x*hit.x + hit.y*hit.y <= surf.semi_aperture*surf.semi_aperture))
        return false;

    // Normal from sphere centre; ensure it opposes the ray
    float invR = 1.0f / fabsf(R);
    norm = dv((hit.x - ctr.x)*invR,
              (hit.y - ctr.y)*invR,
              (hit.z - ctr.z)*invR);
    if (dv_dot(norm, ray.dir) > 0.0f) norm = dv_neg(norm);
    return true;
}

// ---- Refraction / reflection ----

__device__ __forceinline__
bool d_refract(const DVec3& dir, const DVec3& norm,
               float n_ratio, DVec3& out)
{
    float cos_i  = -dv_dot(norm, dir);
    float sin2_t = n_ratio*n_ratio*(1.0f - cos_i*cos_i);
    if (sin2_t >= 1.0f) return false;
    float cos_t  = sqrtf(1.0f - sin2_t);
    float k      = n_ratio*cos_i - cos_t;
    float ox = dir.x*n_ratio + norm.x*k;
    float oy = dir.y*n_ratio + norm.y*k;
    float oz = dir.z*n_ratio + norm.z*k;
    float sq = ox*ox + oy*oy + oz*oz;
    if (sq < 1e-18f || !isfinite(sq)) return false;   // degenerate: discard
    float inv = rsqrtf(sq);
    out = dv(ox*inv, oy*inv, oz*inv);
    return true;
}

__device__ __forceinline__
bool d_reflect(const DVec3& dir, const DVec3& norm, DVec3& out)
{
    float d2 = 2.0f * dv_dot(dir, norm);
    float ox = dir.x - norm.x*d2;
    float oy = dir.y - norm.y*d2;
    float oz = dir.z - norm.z*d2;
    float sq = ox*ox + oy*oy + oz*oz;
    if (sq < 1e-18f || !isfinite(sq)) return false;
    float inv = rsqrtf(sq);
    out = dv(ox*inv, oy*inv, oz*inv);
    return true;
}

// ===========================================================================
// Device-side ghost ray trace
// Mirrors trace_ghost_ray() in trace.cpp exactly.
// ===========================================================================

struct DTraceResult { DVec3 position; float weight; bool valid; };

__device__ __forceinline__
DTraceResult d_trace_ghost_ray(const DRay& ray_in,
                                const Surface* surfs, int n_surfs,
                                float sensor_z,
                                int bounce_a, int bounce_b,
                                float lambda_nm)
{
    DRay  ray          = ray_in;
    float current_ior  = 1.0f;
    float weight       = 1.0f;

    // ---- Phase 1: forward 0..bounce_b (reflect at bounce_b) ----
    for (int s = 0; s <= bounce_b; ++s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfs[s], hit, norm))
            return {dv(0,0,0), 0.0f, false};

        ray.origin = hit;
        float n1   = current_ior;
        float n2   = d_ior_at(surfs[s], lambda_nm);
        float cos_i = fabsf(dv_dot(norm, ray.dir));
        float R     = d_surface_reflectance(cos_i, n1, n2,
                                             surfs[s].coating, lambda_nm);

        if (s == bounce_b)
        {
            DVec3 rdir;
            if (!d_reflect(ray.dir, norm, rdir))
                return {dv(0,0,0), 0.0f, false};
            ray.dir = rdir;
            weight *= R;
        }
        else
        {
            DVec3 new_dir;
            if (!d_refract(ray.dir, norm, n1/n2, new_dir))
                return {dv(0,0,0), 0.0f, false};
            ray.dir     = new_dir;
            weight     *= (1.0f - R);
            current_ior = n2;
        }
    }

    // ---- Phase 2: backward bounce_b-1..bounce_a (reflect at bounce_a) ----
    for (int s = bounce_b - 1; s >= bounce_a; --s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfs[s], hit, norm))
            return {dv(0,0,0), 0.0f, false};

        ray.origin = hit;
        float n1   = current_ior;
        float n2   = d_ior_before(surfs, s, lambda_nm);
        float cos_i = fabsf(dv_dot(norm, ray.dir));
        float R     = d_surface_reflectance(cos_i, n1, n2,
                                             surfs[s].coating, lambda_nm);

        if (s == bounce_a)
        {
            DVec3 rdir;
            if (!d_reflect(ray.dir, norm, rdir))
                return {dv(0,0,0), 0.0f, false};
            ray.dir     = rdir;
            weight     *= R;
            current_ior = d_ior_at(surfs[bounce_a], lambda_nm);
        }
        else
        {
            DVec3 new_dir;
            if (!d_refract(ray.dir, norm, n1/n2, new_dir))
                return {dv(0,0,0), 0.0f, false};
            ray.dir     = new_dir;
            weight     *= (1.0f - R);
            current_ior = n2;
        }
    }

    // ---- Phase 3: forward bounce_a+1..n_surfs-1 ----
    for (int s = bounce_a + 1; s < n_surfs; ++s)
    {
        DVec3 hit, norm;
        if (!d_intersect_surface(ray, surfs[s], hit, norm))
            return {dv(0,0,0), 0.0f, false};

        ray.origin = hit;
        float n1   = current_ior;
        float n2   = d_ior_at(surfs[s], lambda_nm);
        float cos_i = fabsf(dv_dot(norm, ray.dir));
        float R     = d_surface_reflectance(cos_i, n1, n2,
                                             surfs[s].coating, lambda_nm);

        DVec3 new_dir;
        if (!d_refract(ray.dir, norm, n1/n2, new_dir))
            return {dv(0,0,0), 0.0f, false};
        ray.dir     = new_dir;
        weight     *= (1.0f - R);
        current_ior = n2;
    }

    // ---- Propagate to sensor plane ----
    if (fabsf(ray.dir.z) < 1e-12f) return {dv(0,0,0), 0.0f, false};
    float t = (sensor_z - ray.origin.z) / ray.dir.z;
    // Use !(t > 0) rather than (t < 0) so that NaN is caught as invalid.
    if (!(t > 0.0f)) return {dv(0,0,0), 0.0f, false};

    DVec3 pos = dv(ray.origin.x + ray.dir.x*t,
                   ray.origin.y + ray.dir.y*t,
                   ray.origin.z + ray.dir.z*t);
    return {pos, weight, true};
}

// ===========================================================================
// GPU-side data structures (plain POD, no STL)
// ===========================================================================

struct GPUPair   { int surf_a, surf_b; float area_boost; };
struct GPUSource { float angle_x, angle_y, r, g, b; };
struct GPUSample { float u, v; };  // entrance-pupil grid point
// One wavelength sample with pre-computed RGB colour weights.
// Matches GPUSpectralSample in ghost_cuda.h.
struct GPUSpectralSampleDev { float lambda, rw, gw, bw; };

// ===========================================================================
// Scatter kernel
// ===========================================================================

static constexpr int BLOCK_SIZE = 512;

__global__ void ghost_kernel(
    const Surface*  d_surfs,
    int             n_surfs,
    float           sensor_z,
    const GPUPair*  d_pairs,
    int             n_sources,          // needed for index decode
    const GPUSource* d_sources,
    const GPUSample* d_grid,
    int             n_grid,
    float           front_R,
    float           start_z,
    float           sensor_half_w,
    float           sensor_half_h,
    int             width,              // bounding-box buffer width
    int             height,             // bounding-box buffer height
    float*          d_out_r,
    float*          d_out_g,
    float*          d_out_b,
    float           gain,
    float           ray_weight,
    const GPUSpectralSampleDev* d_spec,
    int             n_spec,
    float           fmt_w,              // format width in pixels
    float           fmt_h,              // format height in pixels
    float           fmt_x0_in_buf,      // format origin x within buffer
    float           fmt_y0_in_buf)      // format origin y within buffer
{
    // Decode (pair, source) from flat blockIdx.x
    int ps_idx   = blockIdx.x;
    int pair_idx = ps_idx / n_sources;
    int src_idx  = ps_idx % n_sources;

    // Grid sample index
    int grid_idx = (int)blockIdx.y * BLOCK_SIZE + (int)threadIdx.x;
    if (grid_idx >= n_grid) return;

    const GPUPair&   pair  = d_pairs[pair_idx];

    // Safety: reject pairs with out-of-range surface indices.
    // This can't happen via normal code paths, but guards against any future
    // data-flow bug producing an invalid pair.
    if (pair.surf_a < 0 || pair.surf_b <= pair.surf_a || pair.surf_b >= n_surfs)
        return;
    const GPUSource& src   = d_sources[src_idx];
    const GPUSample& gs    = d_grid[grid_idx];

    // Collimated beam direction for this source (angle → direction)
    float bx = tanf(src.angle_x);
    float by = tanf(src.angle_y);
    DVec3 beam_dir = dv_normalize(dv(bx, by, 1.0f));

    DRay ray;
    ray.origin = dv(gs.u * front_R, gs.v * front_R, start_z);
    ray.dir    = beam_dir;

    for (int s = 0; s < n_spec; ++s)
    {
        const GPUSpectralSampleDev& spec = d_spec[s];

        DTraceResult res = d_trace_ghost_ray(
            ray, d_surfs, n_surfs, sensor_z,
            pair.surf_a, pair.surf_b, spec.lambda);

        if (!res.valid) continue;

        // Guard against non-finite sensor positions (can arise from near-singular
        // ray paths, e.g. ray nearly parallel to optical axis → huge t value).
        if (!isfinite(res.position.x) || !isfinite(res.position.y)) continue;

        float base = res.weight * ray_weight * gain * pair.area_boost;
        if (base < 1e-14f) continue;

        // Map sensor position (mm) → pixel coordinate.
        // Ghost positions are anchored to the format centre, then offset into
        // the bounding-box buffer by fmt_x0_in_buf / fmt_y0_in_buf.  This keeps
        // the optical axis fixed even when off-frame content expands the bbox.
        float px = (res.position.x / (2.0f * sensor_half_w) + 0.5f) * fmt_w + fmt_x0_in_buf;
        float py = (res.position.y / (2.0f * sensor_half_h) + 0.5f) * fmt_h + fmt_y0_in_buf;

        // Guard a second time: dividing by sensor_half_w/h could produce inf/nan
        // if those are zero (degenerate lens / FOV = 0).
        if (!isfinite(px) || !isfinite(py)) continue;

        // Clamp before int conversion to avoid undefined behaviour for out-of-range
        // floats (values outside [INT_MIN, INT_MAX]).  The downstream bounds check
        // will still discard off-screen coordinates; this just ensures defined
        // integer semantics.
        px = fmaxf(-2.0e9f, fminf(2.0e9f, px));
        py = fmaxf(-2.0e9f, fminf(2.0e9f, py));

        int   x0 = (int)floorf(px - 0.5f);
        int   y0 = (int)floorf(py - 0.5f);
        float fx = (px - 0.5f) - (float)x0;
        float fy = (py - 0.5f) - (float)y0;

        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx           * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx           * fy;

        // Accumulate weighted spectral contribution to R, G, B
        float cr = src.r * spec.rw * base;
        float cg = src.g * spec.gw * base;
        float cb = src.b * spec.bw * base;

#define SPLAT_RGB(xi, yi, wt)                                                        \
        if ((xi) >= 0 && (xi) < width && (yi) >= 0 && (yi) < height) {              \
            int _pix = (yi) * width + (xi);                                          \
            if (cr > 1e-14f) atomicAdd(&d_out_r[_pix], cr * (wt));                  \
            if (cg > 1e-14f) atomicAdd(&d_out_g[_pix], cg * (wt));                  \
            if (cb > 1e-14f) atomicAdd(&d_out_b[_pix], cb * (wt));                  \
        }

        SPLAT_RGB(x0,   y0,   w00)
        SPLAT_RGB(x0+1, y0,   w10)
        SPLAT_RGB(x0,   y0+1, w01)
        SPLAT_RGB(x0+1, y0+1, w11)
#undef SPLAT_RGB
    }
}

// ===========================================================================
// CPU launcher
// ===========================================================================

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
    std::string*                    out_error)
{
    if (active_pairs.empty() || sources.empty()) return;

    const int n_surfs  = lens.num_surfaces();
    if (n_surfs <= 0) return;   // no surfaces — nothing to render

    // Friendly GPU availability check — diagnose common "no CUDA" situations
    // before we hit a cryptic allocator error.
    {
        int device_count = 0;
        cudaError_t ce = cudaGetDeviceCount(&device_count);
        if (ce == cudaErrorInsufficientDriver) {
            if (out_error)
                *out_error = "FlareSim requires an NVIDIA GPU with an up-to-date driver. "
                             "The installed CUDA driver is too old — please update your "
                             "NVIDIA driver (520 or newer recommended).";
            return;
        }
        if (ce == cudaErrorNoDevice || device_count == 0) {
            if (out_error)
                *out_error = "FlareSim requires an NVIDIA CUDA GPU — no compatible GPU "
                             "was detected on this system. FlareSim will produce black output.";
            return;
        }
        if (ce != cudaSuccess) {
            if (out_error)
                *out_error = std::string("FlareSim: CUDA initialisation failed (")
                             + cudaGetErrorString(ce)
                             + "). Check that your NVIDIA driver is installed correctly.";
            return;
        }
    }

    // Clear any sticky error left by a previous frame's kernel.
    // Note: on Windows/WDDM, once the CUDA context is lost (e.g. from a GPU
    // fault) there is no in-process recovery — cudaDeviceReset() would also
    // destroy Nuke's own CUDA contexts.  We simply clear the error flag here
    // and let the subsequent cudaMalloc report a fresh failure if the context
    // is genuinely gone.
    {
        cudaError_t prev_err = cudaGetLastError();
        if (prev_err != cudaSuccess) {
            fprintf(stderr,
                    "FlareSim: clearing CUDA sticky error: %s\n",
                    cudaGetErrorString(prev_err));
        }
    }

    const int n_pairs  = (int)active_pairs.size();

    // No internal source cap — the caller (Max Sources knob) is responsible
    // for limiting source count.  gridDim.x supports up to 2^31-1 on all
    // CUDA compute capability >= 3.0 devices (i.e. any GPU since 2012).
    const std::vector<BrightPixel>& capped_sources = sources;
    const int n_src    = (int)capped_sources.size();

    // ---- Build entrance-pupil grid ----
    // Three sampling modes controlled by config.pupil_jitter:
    //   0 = regular NxN grid (cell centres)
    //   1 = stratified jitter (one random sample per cell, stays within cell)
    //   2 = Halton quasi-random low-discrepancy sequence (base-2 / base-3)
    //
    // All modes generate up to N*N candidate positions, apply the circular /
    // polygonal aperture mask, and pack survivors into grid_samples[].
    //
    // Wang hash — fast 32-bit scramble, uniform in [0, 2^32).
    auto wang_hash = [](uint32_t s) -> uint32_t {
        s = (s ^ 61u) ^ (s >> 16u);
        s *= 9u;
        s ^= s >> 4u;
        s *= 0x27d4eb2du;
        s ^= s >> 15u;
        return s;
    };
    // Halton base-2 — bit reversal (O(1) via parallel bit ops).
    auto halton2 = [](uint32_t n) -> float {
        n = (n << 16u) | (n >> 16u);
        n = ((n & 0x00ff00ffu) << 8u) | ((n & 0xff00ff00u) >> 8u);
        n = ((n & 0x0f0f0f0fu) << 4u) | ((n & 0xf0f0f0f0u) >> 4u);
        n = ((n & 0x33333333u) << 2u) | ((n & 0xccccccccu) >> 2u);
        n = ((n & 0x55555555u) << 1u) | ((n & 0xaaaaaaaau) >> 1u);
        return (float)n * (1.0f / 4294967296.0f);
    };
    // Halton base-3 — iterative radical inverse.
    auto halton3 = [](uint32_t n) -> float {
        float r = 0.0f, f = 1.0f / 3.0f;
        while (n > 0) { r += (n % 3u) * f; n /= 3u; f /= 3.0f; }
        return r;
    };

    std::vector<GPUSample> grid_samples;
    {
        const int   N          = config.ray_grid;
        const int   n_blades   = config.aperture_blades;
        const float rot_rad    = config.aperture_rotation_deg * ((float)M_PI / 180.0f);
        const bool  polygonal  = (n_blades >= 3);
        const float apothem    = polygonal ? std::cos((float)M_PI / n_blades) : 1.0f;
        const float sector_ang = polygonal ? (2.0f * (float)M_PI / n_blades)  : 1.0f;
        const int      jitter      = config.pupil_jitter;
        // Mix the per-frame seed into the stratified Wang-hash to animate the
        // noise pattern across frames (temporal AA).  Multiply by a large prime
        // so adjacent integer seeds produce unrelated sample sets.
        const uint32_t seed_offset = (uint32_t)config.pupil_jitter_seed * 1000003u;

        grid_samples.reserve((size_t)N * N);
        for (int k = 0; k < N * N; ++k)
        {
            const int gx = k % N;
            const int gy = k / N;

            float u, v;
            if (jitter == 2)
            {
                // Halton: quasi-random, independent of (gx, gy).
                u = halton2((uint32_t)k) * 2.0f - 1.0f;
                v = halton3((uint32_t)k) * 2.0f - 1.0f;
            }
            else
            {
                // Regular (0.5) or stratified (random in [0,1)), optionally seeded.
                float ju = (jitter == 1) ? wang_hash((uint32_t)k + seed_offset)
                                               / 4294967296.0f : 0.5f;
                float jv = (jitter == 1) ? wang_hash((uint32_t)k + (uint32_t)(N * N) + seed_offset)
                                               / 4294967296.0f : 0.5f;
                u = ((gx + ju) / N) * 2.0f - 1.0f;
                v = ((gy + jv) / N) * 2.0f - 1.0f;
            }

            float r2 = u*u + v*v;
            if (r2 > 1.0f) continue; // outside circumscribed circle
            if (polygonal)
            {
                float angle  = std::atan2(v, u) - rot_rad;
                float sector = std::fmod(angle, sector_ang);
                if (sector < 0.0f) sector += sector_ang;
                if (std::sqrt(r2) * std::cos(sector - sector_ang * 0.5f) > apothem)
                    continue; // outside polygon
            }
            grid_samples.push_back({u, v});
        }
    }
    const int   n_grid     = (int)grid_samples.size();
    if (n_grid == 0) return;  // ray_grid knob set to 0
    const float ray_weight = 1.0f / n_grid;
    const float front_R    = lens.surfaces[0].semi_aperture;
    const float start_z    = lens.surfaces[0].z - 20.0f;

    // ---- Pack GPU-side pair / source arrays ----
    std::vector<GPUPair> gpu_pairs(n_pairs);
    for (int i = 0; i < n_pairs; ++i)
        gpu_pairs[i] = { active_pairs[i].surf_a,
                         active_pairs[i].surf_b,
                         pair_area_boosts[i] };

    std::vector<GPUSource> gpu_src(n_src);
    for (int i = 0; i < n_src; ++i)
        gpu_src[i] = { capped_sources[i].angle_x, capped_sources[i].angle_y,
                       capped_sources[i].r, capped_sources[i].g, capped_sources[i].b };

    // ---- Build spectral sample table ----
    // Each entry has a wavelength and pre-normalised RGB colour weights.
    // With spectral_samples=3 we use the config wavelengths directly and
    // assign one channel per sample (exact backward compat).
    // With more samples we distribute evenly 400–700 nm and use CIE-approximate
    // Gaussian colour matching, normalised so the sum per channel equals 1.
    std::vector<GPUSpectralSampleDev> spectral_cpu;
    {
        const int ns = std::max(3, config.spectral_samples);
        spectral_cpu.resize(ns);

        // Gaussian approximation of CIE 1931 xyz (→ sRGB).
        auto cie_r = [](float l) {
            float a = (l - 600.0f) / 70.0f, b = (l - 450.0f) / 30.0f;
            return std::max(0.0f, 0.63f * std::exp(-0.5f*a*a)
                                + 0.22f * std::exp(-0.5f*b*b));
        };
        auto cie_g = [](float l) {
            float a = (l - 545.0f) / 55.0f;
            return std::max(0.0f, std::exp(-0.5f*a*a));
        };
        auto cie_b = [](float l) {
            float a = (l - 445.0f) / 45.0f;
            return std::max(0.0f, std::exp(-0.5f*a*a));
        };

        if (ns == 3)
        {
            // Exact backward compat: one sample per channel, full weight.
            spectral_cpu[0] = { config.wavelengths[0], 1.0f, 0.0f, 0.0f };
            spectral_cpu[1] = { config.wavelengths[1], 0.0f, 1.0f, 0.0f };
            spectral_cpu[2] = { config.wavelengths[2], 0.0f, 0.0f, 1.0f };
        }
        else
        {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            for (int i = 0; i < ns; ++i)
            {
                float lam = 400.0f + (300.0f * i) / (ns - 1);
                spectral_cpu[i].lambda = lam;
                spectral_cpu[i].rw = cie_r(lam);
                spectral_cpu[i].gw = cie_g(lam);
                spectral_cpu[i].bw = cie_b(lam);
                sum_r += spectral_cpu[i].rw;
                sum_g += spectral_cpu[i].gw;
                sum_b += spectral_cpu[i].bw;
            }
            // Normalise so each channel's weights sum to 1.
            for (int i = 0; i < ns; ++i)
            {
                if (sum_r > 1e-9f) spectral_cpu[i].rw /= sum_r;
                if (sum_g > 1e-9f) spectral_cpu[i].gw /= sum_g;
                if (sum_b > 1e-9f) spectral_cpu[i].bw /= sum_b;
            }
        }
    }
    const int n_spec = (int)spectral_cpu.size();

    const size_t n_px = (size_t)width * height;

    int  grid_blocks = (n_grid + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid_dim((unsigned)(n_pairs * n_src), (unsigned)grid_blocks, 1);

    // ---- Grow device buffers if the current allocation is too small ----
    // On success the allocation is reused; on failure we report and return.
    // The cache retains all valid buffers — the next frame will reuse them.

    // Error reporter (used inline below — not a macro to avoid goto issues).
    auto report = [&](cudaError_t e, const char* site) {
        fprintf(stderr, "FlareSim CUDA error at %s -- %s\n", site, cudaGetErrorString(e));
        if (out_error && out_error->empty())
        {
            char buf[256];
            snprintf(buf, sizeof(buf), "CUDA error at %s -- %s", site, cudaGetErrorString(e));
            *out_error = buf;
        }
    };

    // Generic void* buffer grow.
    auto ensure = [&](void*& ptr, size_t& cap, size_t need, const char* tag) -> bool {
        if (need <= cap) return true;
        cudaFree(ptr);  ptr = nullptr;  cap = 0;
        cudaError_t e = cudaMalloc(&ptr, need);
        if (e != cudaSuccess) { report(e, tag); return false; }
        cap = need;
        return true;
    };

    if (!ensure(cache.d_surfs, cache.surfs_bytes, n_surfs * sizeof(Surface),    "d_surfs")) return;
    if (!ensure(cache.d_pairs, cache.pairs_bytes, n_pairs * sizeof(GPUPair),     "d_pairs")) return;
    if (!ensure(cache.d_src,   cache.src_bytes,   n_src   * sizeof(GPUSource),   "d_src"  )) return;
    if (!ensure(cache.d_grid,  cache.grid_bytes,  n_grid  * sizeof(GPUSample),   "d_grid" )) return;
    if (!ensure(cache.d_spec,  cache.spec_bytes,  n_spec  * sizeof(GPUSpectralSampleDev), "d_spec")) return;

    // Output channels are always the same size; grow all three together.
    if (n_px > cache.out_floats)
    {
        cudaFree(cache.d_out_r);  cache.d_out_r = nullptr;
        cudaFree(cache.d_out_g);  cache.d_out_g = nullptr;
        cudaFree(cache.d_out_b);  cache.d_out_b = nullptr;
        cache.out_floats = 0;

        cudaError_t er = cudaMalloc(&cache.d_out_r, n_px * sizeof(float));
        if (er != cudaSuccess) { report(er, "d_out_r"); return; }
        cudaError_t eg = cudaMalloc(&cache.d_out_g, n_px * sizeof(float));
        if (eg != cudaSuccess) { report(eg, "d_out_g"); return; }
        cudaError_t eb = cudaMalloc(&cache.d_out_b, n_px * sizeof(float));
        if (eb != cudaSuccess) { report(eb, "d_out_b"); return; }
        cache.out_floats = n_px;
    }

    // ---- Upload frame data ----

#define GPU_CHK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { report(_e, #call); return; } \
    } while(0)

    GPU_CHK(cudaMemcpy(cache.d_surfs, lens.surfaces.data(),
                       n_surfs * sizeof(Surface), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(cache.d_pairs, gpu_pairs.data(),
                       n_pairs * sizeof(GPUPair), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(cache.d_src,   gpu_src.data(),
                       n_src   * sizeof(GPUSource), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(cache.d_grid,  grid_samples.data(),
                       n_grid  * sizeof(GPUSample), cudaMemcpyHostToDevice));
    GPU_CHK(cudaMemcpy(cache.d_spec,  spectral_cpu.data(),
                       n_spec  * sizeof(GPUSpectralSampleDev), cudaMemcpyHostToDevice));

    // Zero the output buffers each frame (kernel scatters by addition).
    GPU_CHK(cudaMemset(cache.d_out_r, 0, n_px * sizeof(float)));
    GPU_CHK(cudaMemset(cache.d_out_g, 0, n_px * sizeof(float)));
    GPU_CHK(cudaMemset(cache.d_out_b, 0, n_px * sizeof(float)));

    // ---- Launch ----
    printf("FlareSim CUDA: %d pairs x %d sources x %d samples "
           "-> grid (%u, %u, 1)  block (%d)\n",
           n_pairs, n_src, n_grid,
           grid_dim.x, grid_dim.y, BLOCK_SIZE);
    fflush(stdout);

    ghost_kernel<<<grid_dim, block>>>(
        static_cast<Surface*>(cache.d_surfs), n_surfs, lens.sensor_z,
        static_cast<GPUPair*>(cache.d_pairs), n_src,
        static_cast<GPUSource*>(cache.d_src),
        static_cast<GPUSample*>(cache.d_grid), n_grid,
        front_R, start_z,
        sensor_half_w, sensor_half_h,
        width, height,
        cache.d_out_r, cache.d_out_g, cache.d_out_b,
        config.gain, ray_weight,
        static_cast<GPUSpectralSampleDev*>(cache.d_spec), n_spec,
        (float)fmt_w, (float)fmt_h,
        (float)fmt_x0_in_buf, (float)fmt_y0_in_buf);

    GPU_CHK(cudaDeviceSynchronize());

    // ---- Copy results back to CPU ----
    GPU_CHK(cudaMemcpy(out_r, cache.d_out_r, n_px * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHK(cudaMemcpy(out_g, cache.d_out_g, n_px * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHK(cudaMemcpy(out_b, cache.d_out_b, n_px * sizeof(float), cudaMemcpyDeviceToHost));

#undef GPU_CHK
    // Device buffers remain alive in cache for the next frame.
}

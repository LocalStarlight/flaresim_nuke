// ============================================================================
// ghost.cpp — Ghost pair enumeration and pre-filtering
//
// Provides:
//   enumerate_ghost_pairs() — all C(N,2) bounce pairs for a lens
//   filter_ghost_pairs()    — discards dim/invalid pairs, returns area boosts
//
// The actual rendering is done on GPU in ghost_cuda.cu.
// ============================================================================

#include "ghost.h"
#include "trace.h"

#include <cmath>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Enumerate all ghost pairs: every combination of 2 surfaces.
// ---------------------------------------------------------------------------

std::vector<GhostPair> enumerate_ghost_pairs(const LensSystem &lens)
{
    std::vector<GhostPair> pairs;
    int N = lens.num_surfaces();
    for (int a = 0; a < N; ++a)
        for (int b = a + 1; b < N; ++b)
            pairs.push_back({a, b});
    return pairs;
}

// ---------------------------------------------------------------------------
// Pre-filter: trace a single on-axis ray through each ghost pair.
// Returns the average Fresnel weight across RGB wavelengths.
// ---------------------------------------------------------------------------

static float estimate_ghost_intensity(const LensSystem &lens,
                                      int bounce_a, int bounce_b,
                                      const GhostConfig &config)
{
    // On-axis ray at the centre of the entrance pupil
    Ray ray;
    ray.origin = Vec3f(0, 0, lens.surfaces[0].z - 20.0f);
    ray.dir = Vec3f(0, 0, 1);

    float total = 0;
    for (int ch = 0; ch < 3; ++ch)
    {
        TraceResult r = trace_ghost_ray(ray, lens, bounce_a, bounce_b,
                                        config.wavelengths[ch]);
        if (r.valid)
            total += r.weight;
    }
    return total / 3.0f;
}

// ---------------------------------------------------------------------------
// Estimate the ghost image spread for a bounce pair relative to the sensor.
//
// Traces a coarse grid of on-axis rays across the entrance pupil and
// measures the bounding box of sensor landing positions.  Returns a
// correction factor = ghost_area / sensor_area, clamped to [1, max_boost].
//
// Defocused ghost pairs produce images much larger than the sensor,
// diluting per-pixel brightness.  This correction factor compensates
// for that geometric dilution so all ghost pairs remain visible.
// ---------------------------------------------------------------------------

static float estimate_ghost_spread(const LensSystem &lens,
                                   int bounce_a, int bounce_b,
                                   float sensor_half_w, float sensor_half_h,
                                   const GhostConfig &config,
                                   float *out_ghost_w_mm = nullptr,
                                   float *out_ghost_h_mm = nullptr)
{
    constexpr int G = 8; // coarse grid for spread estimation
    float front_R = lens.surfaces[0].semi_aperture;
    float start_z = lens.surfaces[0].z - 20.0f;

    float min_x = 1e30f, max_x = -1e30f;
    float min_y = 1e30f, max_y = -1e30f;
    int valid_count = 0;

    for (int gy = 0; gy < G; ++gy)
    {
        for (int gx = 0; gx < G; ++gx)
        {
            float u = ((gx + 0.5f) / G) * 2.0f - 1.0f;
            float v = ((gy + 0.5f) / G) * 2.0f - 1.0f;
            if (u * u + v * v > 1.0f)
                continue;

            Ray ray;
            ray.origin = Vec3f(u * front_R, v * front_R, start_z);
            ray.dir = Vec3f(0, 0, 1); // on-axis

            // Use green wavelength for spread estimation
            TraceResult res = trace_ghost_ray(ray, lens, bounce_a, bounce_b,
                                              config.wavelengths[1]);
            if (!res.valid)
                continue;

            min_x = std::min(min_x, res.position.x);
            max_x = std::max(max_x, res.position.x);
            min_y = std::min(min_y, res.position.y);
            max_y = std::max(max_y, res.position.y);
            ++valid_count;
        }
    }

    if (valid_count < 2)
        return 1.0f; // too few hits to estimate

    float ghost_w = std::max(max_x - min_x, 0.01f);
    float ghost_h = std::max(max_y - min_y, 0.01f);
    float sensor_w = 2.0f * sensor_half_w;
    float sensor_h = 2.0f * sensor_half_h;

    if (out_ghost_w_mm) *out_ghost_w_mm = ghost_w;
    if (out_ghost_h_mm) *out_ghost_h_mm = ghost_h;

    float area_ratio = (ghost_w * ghost_h) / (sensor_w * sensor_h);
    return std::clamp(area_ratio, 1.0f, config.max_area_boost);
}

// ---------------------------------------------------------------------------
// filter_ghost_pairs — pre-filter used before the CUDA render pass.
// ---------------------------------------------------------------------------

void filter_ghost_pairs(const LensSystem&       lens,
                        float                   sensor_half_w,
                        float                   sensor_half_h,
                        const GhostConfig&      config,
                        std::vector<GhostPair>& active_pairs_out,
                        std::vector<float>&     area_boosts_out)
{
    auto pairs = enumerate_ghost_pairs(lens);
    active_pairs_out.clear();
    area_boosts_out.clear();

    for (auto& p : pairs)
    {
        float ior_before_a = lens.ior_before(p.surf_a);
        float ior_after_a  = lens.surfaces[p.surf_a].ior;
        float ior_before_b = lens.ior_before(p.surf_b);
        float ior_after_b  = lens.surfaces[p.surf_b].ior;

        if (std::abs(ior_before_a - ior_after_a) < 0.001f ||
            std::abs(ior_before_b - ior_after_b) < 0.001f)
            continue;

        float est = estimate_ghost_intensity(lens, p.surf_a, p.surf_b, config);
        if (est < config.min_intensity)
            continue;

        float boost = 1.0f;
        if (config.ghost_normalize)
            boost = estimate_ghost_spread(lens, p.surf_a, p.surf_b,
                                          sensor_half_w, sensor_half_h, config);
        active_pairs_out.push_back(p);
        area_boosts_out.push_back(boost);
    }
}

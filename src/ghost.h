// ============================================================================
// ghost.h — Ghost reflection enumeration and rendering
// ============================================================================
#pragma once

#include "lens.h"

#include <vector>

// A ghost bounce pair: surfaces where light reflects instead of transmitting.
struct GhostPair
{
    int surf_a; // first bounce surface (closer to front)
    int surf_b; // second bounce surface (closer to sensor)
};

// A bright pixel extracted from the input image.
struct BrightPixel
{
    float angle_x; // horizontal angle from optical axis (radians)
    float angle_y; // vertical angle from optical axis (radians)
    float r, g, b; // HDR intensity
};

// Configuration for the ghost renderer.
struct GhostConfig
{
    int ray_grid = 64;                               // samples per dimension across entrance pupil
    float min_intensity = 1e-7f;                     // skip ghost pairs dimmer than this
    float gain = 1000.0f;                            // ghost intensity multiplier
    float wavelengths[3] = {650.0f, 550.0f, 450.0f}; // R, G, B in nm

    // Per-pair area normalization: boost defocused ghost pairs so they remain
    // visible.  Production renderers (ILM, Weta) use a similar technique.
    bool ghost_normalize = true;   // enable per-pair area correction
    float max_area_boost = 100.0f; // clamp the correction factor

    // Aperture shape (0 = circular; >=3 = regular polygon with N sides).
    int   aperture_blades       = 0;
    float aperture_rotation_deg = 0.0f;

    // Spectral quality (3 = classic R/G/B; 5/7/9/11 = smoother chromatic aberration).
    int spectral_samples = 3;

    // Entrance-pupil sampling pattern.
    // 0 = regular grid (default), 1 = stratified jitter, 2 = Halton quasi-random.
    int pupil_jitter = 0;

    // Per-frame seed mixed into the stratified Wang-hash scramble.
    // Animate this value to get a different noise pattern each frame (useful
    // for temporal anti-aliasing of the jitter pattern).  Ignored when
    // pupil_jitter != 1 (stratified mode).
    int pupil_jitter_seed = 0;
};

// Enumerate all valid ghost bounce pairs for the lens system.
// Returns C(N, 2) pairs where N = number of surfaces.
std::vector<GhostPair> enumerate_ghost_pairs(const LensSystem &lens);

// Pre-filter ghost pairs for a lens system.
//
// Removes pairs whose surfaces have no meaningful IOR contrast, then traces a
// single on-axis probe ray to discard pairs below config.min_intensity.
// Optionally estimates the per-pair ghost spread and returns an area-boost
// factor (see GhostConfig::ghost_normalize).
//
// sensor_half_w/h: half-dimensions of the sensor in mm (focal_length × tan(fov/2)).
void filter_ghost_pairs(const LensSystem&       lens,
                        float                   sensor_half_w,
                        float                   sensor_half_h,
                        const GhostConfig&      config,
                        std::vector<GhostPair>& active_pairs_out,
                        std::vector<float>&     area_boosts_out);


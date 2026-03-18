// ============================================================================
// starburst.h — Fraunhofer diffraction starburst PSF
// ============================================================================
#pragma once

#include "ghost.h"   // BrightPixel
#include <vector>

// Configuration for starburst rendering.
struct StarburstConfig
{
    float gain                  = 0.0f;   // 0 = disabled
    float scale                 = 0.15f;  // PSF radius as fraction of image diagonal
    int   aperture_blades       = 0;      // mirrors GhostConfig; 0 = circular
    float aperture_rotation_deg = 0.0f;
    float wavelengths[3]        = {650.0f, 550.0f, 450.0f};  // R, G, B in nm
};

// Pre-computed Fraunhofer diffraction PSF.
// Single-channel (achromatic aperture), DC at centre, peak normalised to 1.
// Chromatic scaling is applied at render time.
struct StarburstPSF
{
    int                N = 0;  // N×N pixels
    std::vector<float> data;   // N×N, row-major, DC at (N/2, N/2)
    bool empty() const { return N == 0 || data.empty(); }
};

// Compute the diffraction PSF from the aperture shape.
// fft_size must be a power of 2; 512 gives good quality for most uses.
void compute_starburst_psf(const StarburstConfig& cfg, StarburstPSF& psf_out,
                            int fft_size = 512);

// Scatter-accumulate the starburst for every source into the output buffers.
//
// tan_half_h / tan_half_v: tangent of half the horizontal / vertical FOV.
// Used to recover source pixel positions from their stored angles.
//
// Values are ADDED to the output buffers (callers should zero them first).
void render_starburst(const StarburstPSF&              psf,
                      const StarburstConfig&            cfg,
                      const std::vector<BrightPixel>&  sources,
                      float                            tan_half_h,
                      float                            tan_half_v,
                      float*                           out_r,
                      float*                           out_g,
                      float*                           out_b,
                      int                              buf_w,
                      int                              buf_h,
                      int                              fmt_w,
                      int                              fmt_h,
                      int                              fmt_x0_in_buf,
                      int                              fmt_y0_in_buf);

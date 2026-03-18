// ============================================================================
// starburst.cpp — Fraunhofer diffraction starburst / diffraction spike renderer
//
// Physics overview
// ─────────────────
// The far-field (Fraunhofer) diffraction pattern of an aperture is the squared
// magnitude of its 2D Fourier transform:
//
//   PSF(fx, fy) = |FT{ A(x, y) }(fx, fy)|²
//
// For a hard-edged polygonal aperture the straight edges produce bright spikes
// perpendicular to each edge direction.  A regular N-gon with N even gives N
// spikes (opposing pairs share a direction); N odd gives 2N spikes.  A
// circular aperture produces no spikes — only the concentric Airy rings.
//
// The pattern scales linearly with wavelength λ: the same physical aperture
// diffracts red light (650 nm) into a ~18 % wider pattern than green (550 nm)
// and ~44 % wider than blue (450 nm).  Rendering R/G/B with different scale
// factors produces the characteristic blue-centre → red-tip colouration.
//
// Implementation
// ──────────────
// 1. Rasterise the aperture onto an N×N binary mask (same polygon test as the
//    entrance-pupil sampler in ghost_cuda.cu — keeps the two representations
//    in sync).
// 2. 2D FFT via a standard radix-2 Cooley–Tukey row-column decomposition.
// 3. Take |FFT|², apply fftshift (DC to centre), normalise peak to 1.
// 4. At render time, gather-sample the PSF at each output pixel near a source,
//    applying a per-channel λ/λ_ref scale factor to the lookup radius.
// ============================================================================

#include "starburst.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// 1D radix-2 Cooley–Tukey FFT (in-place, forward, n must be power of 2).
// ===========================================================================

static void fft1d(std::complex<float>* data, int n)
{
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i)
    {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    // Butterfly stages
    for (int len = 2; len <= n; len <<= 1)
    {
        float ang = -2.0f * (float)M_PI / len;
        std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len)
        {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j)
            {
                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + len / 2] * w;
                data[i + j]           = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// ===========================================================================
// 2D FFT (in-place, row-major N×N, N must be power of 2).
// ===========================================================================

static void fft2d(std::vector<std::complex<float>>& grid, int N)
{
    // Row transforms
    for (int r = 0; r < N; ++r)
        fft1d(grid.data() + (size_t)r * N, N);

    // Column transforms via scratch buffer
    std::vector<std::complex<float>> col(N);
    for (int c = 0; c < N; ++c)
    {
        for (int r = 0; r < N; ++r) col[r] = grid[(size_t)r * N + c];
        fft1d(col.data(), N);
        for (int r = 0; r < N; ++r) grid[(size_t)r * N + c] = col[r];
    }
}

// ===========================================================================
// Aperture mask rasterisation.
// Uses the same circular / polygonal test as ghost_cuda.cu so the starburst
// always matches the aperture shape used for ghost tracing.
// ===========================================================================

static void build_aperture_mask(std::vector<float>& mask, int N,
                                 int n_blades, float rot_deg)
{
    mask.assign((size_t)N * N, 0.0f);

    const float rot_rad    = rot_deg * (float)M_PI / 180.0f;
    const bool  polygonal  = (n_blades >= 3);
    const float apothem    = polygonal ? std::cos((float)M_PI / n_blades) : 1.0f;
    const float sector_ang = polygonal ? (2.0f * (float)M_PI / n_blades) : 1.0f;

    for (int r = 0; r < N; ++r)
    {
        for (int c = 0; c < N; ++c)
        {
            // Map pixel centre to UV space [-1, 1]
            float u  = ((c + 0.5f) / N) * 2.0f - 1.0f;
            float v  = ((r + 0.5f) / N) * 2.0f - 1.0f;
            float r2 = u * u + v * v;

            if (r2 > 1.0f) continue;
            if (polygonal)
            {
                float angle  = std::atan2(v, u) - rot_rad;
                float sector = std::fmod(angle, sector_ang);
                if (sector < 0.0f) sector += sector_ang;
                if (std::sqrt(r2) * std::cos(sector - sector_ang * 0.5f) > apothem)
                    continue;
            }
            mask[(size_t)r * N + c] = 1.0f;
        }
    }
}

// ===========================================================================
// compute_starburst_psf
// ===========================================================================

void compute_starburst_psf(const StarburstConfig& cfg, StarburstPSF& psf_out,
                            int fft_size)
{
    const int N = fft_size;
    psf_out.N   = N;
    psf_out.data.resize((size_t)N * N);

    // 1. Rasterise aperture mask
    std::vector<float> mask;
    build_aperture_mask(mask, N, cfg.aperture_blades, cfg.aperture_rotation_deg);

    // 2. Load into complex array
    std::vector<std::complex<float>> grid((size_t)N * N);
    for (size_t i = 0; i < (size_t)N * N; ++i)
        grid[i] = {mask[i], 0.0f};

    // 3. 2D FFT
    fft2d(grid, N);

    // 4. Power spectrum |FFT|², with simultaneous fftshift.
    //    The fftshift moves the DC component from grid[0][0] to the centre
    //    of the output by reading with modular (half-period offset) indexing:
    //    psf[r][c] = |grid[(r+N/2) % N][(c+N/2) % N]|²
    float peak = 0.0f;
    for (int r = 0; r < N; ++r)
    {
        const int rs = (r + N / 2) % N;
        for (int c = 0; c < N; ++c)
        {
            const auto& z = grid[(size_t)rs * N + (c + N / 2) % N];
            float val = z.real() * z.real() + z.imag() * z.imag();
            psf_out.data[(size_t)r * N + c] = val;
            if (val > peak) peak = val;
        }
    }

    // 5. Normalise so the peak (Airy disk centre) = 1.0.
    //    Per-source gain is applied at render time.
    if (peak > 0.0f)
    {
        const float inv = 1.0f / peak;
        for (float& v : psf_out.data) v *= inv;
    }
}

// ===========================================================================
// render_starburst
//
// Chromatic dispersion: the diffraction pattern scales linearly with λ.
// Red light (650 nm) gets a 650/550 ≈ 1.18× wider pattern than the 550 nm
// reference; blue (450 nm) gets 450/550 ≈ 0.82× narrower.  Sampling the same
// PSF with a channel-dependent scale factor produces the physical colouration:
//   • Centre of spike: all three channels contribute → white
//   • Mid spike: red & green contribute; blue has faded → warm
//   • Tip of spike: only red reaches → red
// ===========================================================================

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
                      int                              fmt_y0_in_buf)
{
    if (psf.empty() || cfg.gain <= 0.0f || sources.empty()) return;

    const int   N      = psf.N;
    const float half_N = N * 0.5f;
    const float diag   = std::sqrt((float)buf_w * buf_w + (float)buf_h * buf_h);
    const float r_ref  = cfg.scale * diag;   // PSF radius at 550 nm reference

    // Wavelength scale factors relative to 550 nm.
    // Red > 1 → larger footprint; blue < 1 → smaller footprint.
    const float kRefLam     = 550.0f;
    const float scale_ch[3] = {
        cfg.wavelengths[0] / kRefLam,   // R: ~1.18
        cfg.wavelengths[1] / kRefLam,   // G: ~1.00
        cfg.wavelengths[2] / kRefLam    // B: ~0.82
    };

    // Format centre in buffer coordinates.
    // Used to invert the angle → pixel mapping from BrightPixel construction:
    //   angle_x = atan(ndc_x * 2 * tan_half_h),  ndc_x = (cx - fmt_cx) / fmt_w
    // Inverse: ndc_x = tan(angle_x) / (2 * tan_half_h)
    //          cx    = fmt_cx + ndc_x * fmt_w
    const float fmt_cx = fmt_x0_in_buf + fmt_w * 0.5f;
    const float fmt_cy = fmt_y0_in_buf + fmt_h * 0.5f;

    float* out_ch[3] = { out_r, out_g, out_b };

    for (const BrightPixel& src : sources)
    {
        // Source pixel position in buffer coordinates
        const float src_px = fmt_cx + (std::tan(src.angle_x) / (2.0f * tan_half_h)) * fmt_w;
        const float src_py = fmt_cy + (std::tan(src.angle_y) / (2.0f * tan_half_v)) * fmt_h;

        // Bounding box using the largest channel radius (red)
        const float r_max = r_ref * scale_ch[0];
        const int   ox0   = std::max(0,     (int)std::floor(src_px - r_max));
        const int   ox1   = std::min(buf_w, (int)std::ceil (src_px + r_max) + 1);
        const int   oy0   = std::max(0,     (int)std::floor(src_py - r_max));
        const int   oy1   = std::min(buf_h, (int)std::ceil (src_py + r_max) + 1);

        if (ox0 >= ox1 || oy0 >= oy1) continue;

        const float src_rgb[3] = { src.r, src.g, src.b };
        const float* p = psf.data.data();

        for (int ch = 0; ch < 3; ++ch)
        {
            // inv_r converts an output-pixel offset from source → PSF grid index.
            // At scale s, a pixel offset d maps to PSF index half_N + d*(half_N/(r_ref*s)).
            const float inv_r = half_N / (r_ref * scale_ch[ch]);
            const float src_v = src_rgb[ch] * cfg.gain;
            float* o          = out_ch[ch];

            for (int oy = oy0; oy < oy1; ++oy)
            {
                const float dj_f = (oy - src_py) * inv_r + half_N;
                if (dj_f < 0.0f || dj_f >= (float)(N - 1)) continue;
                const int   dj = (int)dj_f;
                const float fj = dj_f - dj;

                for (int ox = ox0; ox < ox1; ++ox)
                {
                    const float di_f = (ox - src_px) * inv_r + half_N;
                    if (di_f < 0.0f || di_f >= (float)(N - 1)) continue;
                    const int   di = (int)di_f;
                    const float fi = di_f - di;

                    // Bilinear sample from PSF
                    float val = p[(size_t)dj       * N + di    ] * (1.0f - fi) * (1.0f - fj)
                              + p[(size_t)dj       * N + di + 1] *         fi  * (1.0f - fj)
                              + p[(size_t)(dj + 1) * N + di    ] * (1.0f - fi) *         fj
                              + p[(size_t)(dj + 1) * N + di + 1] *         fi  *         fj;

                    o[(size_t)oy * buf_w + ox] += src_v * val;
                }
            }
        }
    }
}

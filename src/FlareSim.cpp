// ============================================================================
// FlareSim.cpp — Nuke Iop plugin for physically-based lens flare simulation
//
// Architecture (lazy-compute-in-engine):
//   _validate()  — sets up channels/format, loads lens, pre-clears buffers,
//                  sets needs_compute_ = true.
//   engine()     — on the first scanline call, acquires compute_mutex_ and
//                  runs the full tile read + CUDA simulation (do_compute()).
//                  Subsequent scanline calls read directly from the cache.
//
// This design is required because Nuke's background playback cacher calls
// _validate() before input pixels are available. By deferring pixel reads to
// engine() we guarantee the input is fully rendered before we touch it.
// ============================================================================

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Channel.h"
#include "DDImage/ChannelSet.h"

#include "lens.h"
#include "ghost.h"
#include "ghost_cuda.h"
#include "starburst.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace DD::Image;

// ---------------------------------------------------------------------------
// Named output channels
// ---------------------------------------------------------------------------

static Channel kFlareRed,  kFlareGreen,  kFlareBlue;
static Channel kSourceRed, kSourceGreen, kSourceBlue;
static Channel kHazeRed,       kHazeGreen,       kHazeBlue;
static Channel kStarburstRed,  kStarburstGreen,  kStarburstBlue;

static void init_channels()
{
    static bool done = false;
    if (done) return;
    kFlareRed    = getChannel("flare.red");
    kFlareGreen  = getChannel("flare.green");
    kFlareBlue   = getChannel("flare.blue");
    kSourceRed   = getChannel("source.red");
    kSourceGreen = getChannel("source.green");
    kSourceBlue  = getChannel("source.blue");
    kHazeRed          = getChannel("haze.red");
    kHazeGreen        = getChannel("haze.green");
    kHazeBlue         = getChannel("haze.blue");
    kStarburstRed     = getChannel("starburst.red");
    kStarburstGreen   = getChannel("starburst.green");
    kStarburstBlue    = getChannel("starburst.blue");
    done = true;
}

// ---------------------------------------------------------------------------
// Sensor size presets
// ---------------------------------------------------------------------------

struct SensorPreset { const char* name; float w, h; };
static const SensorPreset kSensorPresets[] = {
    { "Custom",            0.0f,   0.0f  },
    { "Full Frame",       36.0f,  24.0f  },
    { "Super 35",         24.89f, 18.66f },
    { "APS-C Canon",      22.3f,  14.9f  },
    { "APS-C Nikon/Sony", 23.5f,  15.6f  },
    { "Micro Four Thirds",17.3f,  13.0f  },
    { nullptr, 0, 0 }
};
static const char* const kPresetNames[] = {
    "Custom", "Full Frame", "Super 35",
    "APS-C Canon", "APS-C Nikon/Sony", "Micro Four Thirds", nullptr
};

// ---------------------------------------------------------------------------
// Separable box blur (prefix-sum, O(w*h) per pass).
// Blurs buf in-place; tmp is a scratch buffer (resized as needed).
// ---------------------------------------------------------------------------

static void box_blur(float* buf, int w, int h, int radius, int passes,
                     std::vector<float>& tmp)
{
    if (radius < 1 || passes < 1) return;
    const size_t npx = (size_t)w * h;
    tmp.resize(npx);

    const int maxdim = std::max(w, h);
    std::vector<double> ps(maxdim + 1);

    for (int p = 0; p < passes; ++p)
    {
        // ---- Horizontal pass: buf -> tmp ----
        for (int y = 0; y < h; ++y)
        {
            const float* row = buf + (size_t)y * w;
            float*       out = tmp.data() + (size_t)y * w;
            ps[0] = 0.0;
            for (int x = 0; x < w; ++x)
                ps[x + 1] = ps[x] + row[x];
            for (int x = 0; x < w; ++x)
            {
                int lo = std::max(0,     x - radius);
                int hi = std::min(w - 1, x + radius);
                out[x] = (float)((ps[hi + 1] - ps[lo]) / (hi - lo + 1));
            }
        }
        std::memcpy(buf, tmp.data(), npx * sizeof(float));

        // ---- Vertical pass: buf -> tmp ----
        for (int x = 0; x < w; ++x)
        {
            ps[0] = 0.0;
            for (int y = 0; y < h; ++y)
                ps[y + 1] = ps[y] + buf[(size_t)y * w + x];
            for (int y = 0; y < h; ++y)
            {
                int lo = std::max(0,     y - radius);
                int hi = std::min(h - 1, y + radius);
                tmp[(size_t)y * w + x] = (float)((ps[hi + 1] - ps[lo]) / (hi - lo + 1));
            }
        }
        std::memcpy(buf, tmp.data(), npx * sizeof(float));
    }
}

// ===========================================================================
// Per-pair toggle knob name storage
// ===========================================================================

static constexpr int MAX_PAIRS_UI = 500;

struct PairKnobNames
{
    char names[MAX_PAIRS_UI][16];
    PairKnobNames()
    {
        for (int i = 0; i < MAX_PAIRS_UI; ++i)
            snprintf(names[i], sizeof(names[i]), "pair_%d", i);
    }
};
static PairKnobNames s_pair_knob_names;

// ---------------------------------------------------------------------------
// cluster_sources — merge nearby bright sources into fewer, stronger ones.
//
// Sources within radius_px pixels of the cluster seed are absorbed into it.
// Position becomes the luma-weighted centroid; RGB values are summed so the
// merged cluster carries the combined energy of all constituent sources.
//
// Algorithm: greedy single-pass (O(n²)).  Seeds are taken in descending-luma
// order so the brightest source always anchors each cluster.  n is typically
// small (tens to low hundreds), so O(n²) is negligible.
// ---------------------------------------------------------------------------

static void cluster_sources(std::vector<BrightPixel>& sources,
                            int   radius_px,
                            int   fmt_w,
                            float tan_half_h)
{
    if (radius_px <= 0 || (int)sources.size() < 2) return;

    // 1 pixel ≈ 2*tan_half_h / fmt_w radians at the image centre.
    // Accurate enough for clustering; extreme-angle parallax is irrelevant here.
    const float ang_per_px = 2.0f * tan_half_h / (float)fmt_w;
    const float thresh     = (float)radius_px * ang_per_px;
    const float thresh_sq  = thresh * thresh;

    // Sort descending luma so we seed on the brightest source.
    std::sort(sources.begin(), sources.end(),
              [](const BrightPixel& a, const BrightPixel& b) {
                  return (0.2126f*a.r + 0.7152f*a.g + 0.0722f*a.b)
                       > (0.2126f*b.r + 0.7152f*b.g + 0.0722f*b.b);
              });

    const int n = (int)sources.size();
    std::vector<bool> consumed(n, false);
    std::vector<BrightPixel> out;
    out.reserve(n);

    for (int i = 0; i < n; ++i)
    {
        if (consumed[i]) continue;

        const BrightPixel& seed = sources[i];
        float luma_i = 0.2126f*seed.r + 0.7152f*seed.g + 0.0722f*seed.b;
        float sum_w  = luma_i;
        float ax = seed.angle_x * sum_w;
        float ay = seed.angle_y * sum_w;
        float r  = seed.r, g = seed.g, b = seed.b;

        for (int j = i + 1; j < n; ++j)
        {
            if (consumed[j]) continue;
            const BrightPixel& t = sources[j];
            const float dx = seed.angle_x - t.angle_x;
            const float dy = seed.angle_y - t.angle_y;
            if (dx*dx + dy*dy <= thresh_sq) {
                const float luma_j = 0.2126f*t.r + 0.7152f*t.g + 0.0722f*t.b;
                r += t.r;  g += t.g;  b += t.b;
                ax    += t.angle_x * luma_j;
                ay    += t.angle_y * luma_j;
                sum_w += luma_j;
                consumed[j] = true;
            }
        }

        BrightPixel merged;
        merged.angle_x = (sum_w > 0.0f) ? ax / sum_w : seed.angle_x;
        merged.angle_y = (sum_w > 0.0f) ? ay / sum_w : seed.angle_y;
        merged.r = r;  merged.g = g;  merged.b = b;
        out.push_back(merged);
    }

    sources = std::move(out);
}

// ---------------------------------------------------------------------------
// FlareSim
// ---------------------------------------------------------------------------

class FlareSim : public Iop
{
public:
    // ---- Knob storage ----
    const char* lens_file_;
    float       flare_gain_;
    int         ray_grid_;
    float       threshold_;
    float       source_cap_;       // 0 = off
    int         max_sources_;      // 0 = unlimited
    int         downsample_;
    int         cluster_radius_;   // pixels; 0 = off
    bool        fov_use_sensor_;
    int         sensor_preset_;
    float       fov_h_deg_;
    float       fov_v_deg_;
    bool        fov_auto_v_;
    float       sensor_w_mm_;
    float       sensor_h_mm_;
    float       focal_length_mm_;
    float       ghost_blur_;
    int         ghost_blur_passes_;

    // Haze / veiling glare
    float       haze_gain_;        // 0 = off
    float       haze_radius_;      // fraction of diagonal
    int         haze_blur_passes_;

    // Starburst / diffraction spikes
    float       starburst_gain_;   // 0 = off
    float       starburst_scale_;  // PSF radius as fraction of diagonal

    // Aperture
    int         aperture_blades_;
    float       aperture_rotation_;

    // Pupil sampling
    int         pupil_jitter_;
    int         jitter_seed_;
    bool        jitter_auto_seed_; // derive seed from frame number automatically

    // Spectral quality (index into {3,5,7,9,11})
    int         spectral_idx_;

    // Source preview
    bool        show_sources_;

    // Output mode (0=separate, 1=flare as RGB, 2=sources as RGB)
    int         output_mode_;

    // Per-pair toggles (MAX_PAIRS_UI pre-allocated bools)
    bool        pair_enabled_[MAX_PAIRS_UI];
    // Per-instance labels for pair Bool_knobs (stable pointers for Knob API)
    char        pair_labels_[MAX_PAIRS_UI][64];

    // ---- Runtime state ----
    LensSystem  lens_;
    std::string last_lens_file_;

    // Full enumerated pair list — updated when lens changes.
    std::vector<GhostPair> all_lens_pairs_;

    // Diagnostic counters — written atomically by do_compute().
    std::atomic<int> last_src_count_  {0};
    std::atomic<int> last_pair_count_ {0};

    // Ghost / source / haze / starburst accumulation buffers (CPU, width×height).
    std::vector<float> ghost_r_, ghost_g_, ghost_b_;
    std::vector<float> source_r_, source_g_, source_b_;
    std::vector<float> haze_r_, haze_g_, haze_b_;
    std::vector<float> starburst_r_, starburst_g_, starburst_b_;

    // Starburst PSF cache — recomputed only when aperture shape changes.
    StarburstPSF starburst_psf_;
    int          last_sb_blades_   = -1;
    float        last_sb_rotation_ = 0.0f;
    int cache_width_  = 0;
    int cache_height_ = 0;

    // Persistent GPU buffer cache — avoids per-frame cudaMalloc/cudaFree.
    // Only accessed inside do_compute() which runs under compute_mutex_.
    GpuBufferCache gpu_cache_;

    // CUDA error from last do_compute() — written under compute_mutex_.
    std::string pending_cuda_error_;

    // Lazy-compute state (all written under compute_mutex_).
    bool        needs_compute_ = false;
    int         pending_x0_    = 0;
    int         pending_y0_    = 0;
    int         pending_x1_    = 0;
    int         pending_y1_    = 0;
    int         pending_w_     = 0;
    int         pending_h_     = 0;
    // Format dimensions — anchor optical axis to format centre, not bbox.
    int         pending_fmt_x0_ = 0;
    int         pending_fmt_y0_ = 0;
    int         pending_fmt_w_  = 0;
    int         pending_fmt_h_  = 0;
    // Snapshot of all_lens_pairs_ for do_compute() (safe from _validate() race).
    std::vector<GhostPair> pending_all_pairs_;
    std::mutex  compute_mutex_;

    // ---- Constructor ----
    explicit FlareSim(Node* node)
        : Iop(node)
        , lens_file_("")
        , flare_gain_(1000.0f)
        , ray_grid_(64)
        , threshold_(2.0f)
        , source_cap_(0.0f)
        , max_sources_(512)
        , downsample_(4)
        , cluster_radius_(0)
        , fov_use_sensor_(false)
        , sensor_preset_(0)
        , fov_h_deg_(40.0f)
        , fov_v_deg_(24.0f)
        , fov_auto_v_(true)
        , sensor_w_mm_(36.0f)
        , sensor_h_mm_(24.0f)
        , focal_length_mm_(50.0f)
        , ghost_blur_(0.003f)
        , ghost_blur_passes_(3)
        , haze_gain_(0.0f)
        , haze_radius_(0.15f)
        , haze_blur_passes_(3)
        , starburst_gain_(0.0f)
        , starburst_scale_(0.15f)
        , aperture_blades_(0)
        , aperture_rotation_(0.0f)
        , pupil_jitter_(0)
        , jitter_seed_(0)
        , jitter_auto_seed_(true)
        , spectral_idx_(0)
        , show_sources_(false)
        , output_mode_(0)
    {
        for (int k = 0; k < MAX_PAIRS_UI; ++k)
            pair_enabled_[k] = true;
        for (int k = 0; k < MAX_PAIRS_UI; ++k)
            snprintf(pair_labels_[k], sizeof(pair_labels_[k]), "Pair %d", k);
        init_channels();
    }

    // ---- Nuke boilerplate ----
    const char* Class()     const override { return "FlareSim"; }
    const char* node_help() const override
    {
        return "Physically-based lens flare simulation (ghost ray tracing).\n\n"
               "Loads a .lens prescription file and traces ghost reflections "
               "for every bright pixel in the input image.\n\n"
               "Output channels:\n"
               "  flare.rgb   — ghost reflections\n"
               "  source.rgb  — source detection map (requires Show Sources)\n"
               "  haze.rgb    — veiling glare (requires Haze Gain > 0)\n\n"
               "Connect a second input to use it as a mask (alpha channel) "
               "that gates which bright regions drive the flare.\n\n"
               "Merge the flare over the beauty with a Merge (plus) node.";
    }

    static const Iop::Description d;

    int  maximum_inputs() const override { return 2; }
    int  minimum_inputs() const override { return 1; }
    const char* input_label(int idx, char*) const override
    {
        if (idx == 1) return "mask";
        return "";
    }

    // ---- rebuild_pair_ui ----
    // Runs the physics filter with current knob values to determine which pairs
    // produce a visible contribution, then updates the Pairs tab to show only
    // those pairs.  Populates all_lens_pairs_ as a side-effect so do_compute()
    // can use it for toggle matching.
    void rebuild_pair_ui()
    {
        if (lens_.surfaces.empty()) {
            all_lens_pairs_.clear();
            for (int k = 0; k < MAX_PAIRS_UI; ++k) {
                Knob* kn = knob(s_pair_knob_names.names[k]);
                if (kn) kn->set_flag(Knob::HIDDEN);
            }
            return;
        }

        // Compute approximate sensor half-dimensions from current knob values.
        float fov_h, fov_v;
        if (fov_use_sensor_) {
            const float fl = std::max(focal_length_mm_, 0.1f);
            fov_h = 2.0f * std::atan(sensor_w_mm_ / (2.0f * fl));
            fov_v = 2.0f * std::atan(sensor_h_mm_ / (2.0f * fl));
        } else {
            fov_h = (float)(fov_h_deg_ * M_PI / 180.0);
            // Use captured format aspect if available, otherwise assume 16:9.
            const float aspect = (pending_fmt_w_ > 0 && pending_fmt_h_ > 0)
                                 ? (float)pending_fmt_w_ / pending_fmt_h_
                                 : 16.0f / 9.0f;
            fov_v = fov_auto_v_
                ? 2.0f * std::atan(std::tan(fov_h * 0.5f) / aspect)
                : (float)(fov_v_deg_ * M_PI / 180.0);
        }
        const float shw = lens_.focal_length * std::tan(fov_h * 0.5f);
        const float shh = lens_.focal_length * std::tan(fov_v * 0.5f);

        GhostConfig probe_cfg;  // defaults are fine; only min_intensity matters
        std::vector<GhostPair> active_pairs;
        std::vector<float>     boosts;
        filter_ghost_pairs(lens_, shw, shh, probe_cfg, active_pairs, boosts);

        // If filtering returned nothing (e.g. degenerate settings), fall back
        // to the full enumerated list so the tab is never blank.
        if (active_pairs.empty())
            active_pairs = enumerate_ghost_pairs(lens_);

        if ((int)active_pairs.size() > MAX_PAIRS_UI) {
            fprintf(stderr, "FlareSim: %d active pairs, displaying first %d in Pairs tab.\n",
                    (int)active_pairs.size(), MAX_PAIRS_UI);
            active_pairs.resize(MAX_PAIRS_UI);
        }

        all_lens_pairs_ = active_pairs;

        const int n = (int)all_lens_pairs_.size();
        for (int k = 0; k < n; ++k)
        {
            const GhostPair& p = all_lens_pairs_[k];
            snprintf(pair_labels_[k], sizeof(pair_labels_[k]),
                     "Pair %d  (surf %d <-> surf %d)",
                     k, p.surf_a, p.surf_b);

            Knob* kn = knob(s_pair_knob_names.names[k]);
            if (!kn) continue;
            kn->label(pair_labels_[k]);
            kn->clear_flag(Knob::HIDDEN);
        }
        // Hide slots beyond the active count
        for (int k = n; k < MAX_PAIRS_UI; ++k) {
            Knob* kn = knob(s_pair_knob_names.names[k]);
            if (kn) kn->set_flag(Knob::HIDDEN);
        }

        printf("FlareSim: %d active ghost pair(s) shown in Pairs tab.\n", n);
    }

    // ---- Knobs ----
    void knobs(Knob_Callback f) override
    {
        File_knob(f, &lens_file_,  "lens_file",      "Lens File");
        Tooltip(f, "Path to a .lens prescription file.");

        Divider(f, "Ghost");
        Float_knob(f, &flare_gain_, "flare_gain",    "Flare Gain");
        Tooltip(f, "Ghost intensity multiplier.");
        Int_knob(f,   &ray_grid_,   "ray_grid",      "Ray Grid (NxN)");
        Tooltip(f, "NxN entrance-pupil samples per source. "
                   "Higher = smoother ghosts, longer render time.");

        static const char* const kJitterModes[] = {
            "Off", "Stratified", "Halton", nullptr
        };
        Enumeration_knob(f, &pupil_jitter_, kJitterModes, "pupil_jitter", "Pupil Jitter");
        Tooltip(f, "Entrance-pupil sampling pattern.\n"
                   "Off: regular NxN grid — fastest, but visible dot pattern at low ray counts.\n"
                   "Stratified: one randomly-placed sample per grid cell — breaks up the dot pattern.\n"
                   "Halton: quasi-random low-discrepancy sequence — deterministic, very even coverage.");

        Int_knob(f, &jitter_seed_, "jitter_seed", "Jitter Seed");
        Tooltip(f, "Integer seed for the stratified noise pattern. "
                   "Only used when Auto Seed is off and Pupil Jitter is Stratified.");
        Bool_knob(f, &jitter_auto_seed_, "jitter_auto_seed", "Auto Seed");
        Tooltip(f, "When on (the default), the jitter seed is automatically set to the current "
                   "frame number, so sample positions change on every frame without any manual "
                   "keyframing — equivalent to animating Jitter Seed = frame.\n\n"
                   "Turn off to use the fixed Jitter Seed value, which gives a repeatable "
                   "pattern regardless of frame number (useful for still renders or when you "
                   "need the exact same noise across multiple passes).\n\n"
                   "Only affects Stratified mode; ignored for Off and Halton.");

        Float_knob(f, &threshold_,  "threshold",     "Threshold");
        Tooltip(f, "Minimum luminance for a pixel to count as a bright source.");
        Float_knob(f, &source_cap_, "source_cap",    "Source Cap");
        Tooltip(f, "Maximum source luminance before contribution to ghosts. "
                   "0 = off (unlimited). Useful to prevent very bright sources "
                   "from overpowering the flare.");
        Int_knob(f, &max_sources_, "max_sources",   "Max Sources");
        Tooltip(f, "Maximum number of bright sources passed to the GPU per frame. "
                   "The N brightest sources are kept; the rest are discarded. "
                   "0 = unlimited (uses a hard internal safety limit). "
                   "Reduce this if you experience GPU crashes on frames with many sources.");
        Int_knob(f,   &downsample_, "downsample",    "Downsample");
        Tooltip(f, "Sample every Nth pixel in each dimension when extracting bright sources. "
                   "4 means 1/16th of pixels are used. Lower = more sources = brighter but slower.");
        Int_knob(f, &cluster_radius_, "cluster_radius", "Cluster Radius");
        Tooltip(f, "Merge nearby bright sources that fall within this radius (in pixels) of each other "
                   "into a single, stronger source at their luma-weighted centroid.\n\n"
                   "Useful when a large bright area (e.g. a car headlight, the sun, or a bokeh highlight) "
                   "is detected as many adjacent sources that produce near-identical, overlapping ghosts. "
                   "Clustering reduces GPU work and eliminates the slight banding that those duplicate "
                   "ghosts can produce.\n\n"
                   "0 = off. A value roughly equal to the physical size of your brightest highlights "
                   "in pixels is a good starting point (e.g. 20-50 for a typical headlight).");

        Divider(f, "Camera");
        Bool_knob(f, &fov_use_sensor_, "fov_use_sensor", "Use Sensor Size");
        Tooltip(f, "When enabled, FOV is computed from sensor dimensions and focal length.");
        Enumeration_knob(f, &sensor_preset_, kPresetNames, "sensor_preset", "Sensor Preset");
        if (!fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Tooltip(f, "Common sensor size presets. Selecting a preset fills Sensor Width and Height.");

        Float_knob(f, &fov_h_deg_,  "fov_h",         "FOV H (deg)");
        SetRange(f, 1.0, 180.0);
        if (fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Tooltip(f, "Horizontal field of view of the input image in degrees.");
        Bool_knob(f, &fov_auto_v_,  "fov_auto_v",    "Auto FOV V");
        if (fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Tooltip(f, "Derive vertical FOV from horizontal FOV and image aspect ratio.");
        Float_knob(f, &fov_v_deg_,  "fov_v",         "FOV V (deg)");
        if (fov_auto_v_ || fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Tooltip(f, "Vertical field of view in degrees. Ignored when Auto FOV V is enabled.");

        Float_knob(f, &sensor_w_mm_,      "sensor_w",      "Sensor Width (mm)");
        SetRange(f, 1.0, 100.0);
        if (!fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Float_knob(f, &sensor_h_mm_,      "sensor_h",      "Sensor Height (mm)");
        SetRange(f, 1.0, 100.0);
        if (!fov_use_sensor_) SetFlags(f, Knob::DISABLED);
        Float_knob(f, &focal_length_mm_,  "focal_length",  "Focal Length (mm)");
        SetRange(f, 1.0, 2000.0);
        if (!fov_use_sensor_) SetFlags(f, Knob::DISABLED);

        Divider(f, "Aperture");
        Int_knob(f, &aperture_blades_, "aperture_blades", "Aperture Blades");
        SetRange(f, 0.0, 16.0);
        Tooltip(f, "0 = circular aperture. 3–16 = regular polygon — affects ghost shape.");
        Float_knob(f, &aperture_rotation_, "aperture_rotation", "Aperture Rotation");
        SetRange(f, -180.0, 180.0);
        Tooltip(f, "Rotation of the aperture polygon in degrees.");

        Divider(f, "Spectral");
        static const char* const kSpecNames[] = { "3 (R/G/B)", "5", "7", "9", "11", nullptr };
        Enumeration_knob(f, &spectral_idx_, kSpecNames, "spectral_samples", "Spectral Samples");
        Tooltip(f, "Number of wavelength samples for chromatic aberration. "
                   "3 = classic R/G/B (fastest). Higher = smoother colour fringing.");

        Divider(f, "Post-process");
        Float_knob(f, &ghost_blur_,        "ghost_blur",        "Ghost Blur");
        Tooltip(f, "Post-splat blur radius as a fraction of the image diagonal. 0 = off.");
        Int_knob(f,   &ghost_blur_passes_, "ghost_blur_passes", "Ghost Blur Passes");
        Tooltip(f, "Number of box-blur passes. 3 approximates a Gaussian.");

        Float_knob(f, &haze_gain_,        "haze_gain",        "Haze Gain");
        Tooltip(f, "Veiling glare intensity multiplier. 0 = off. "
                   "Haze is a wide soft glow splatted from each bright source into haze.rgb. "
                   "Useful for simulating scattered light inside the lens barrel.");
        Float_knob(f, &haze_radius_,      "haze_radius",      "Haze Radius");
        Tooltip(f, "Haze blur radius as a fraction of the image diagonal. "
                   "Larger values produce a wider, softer glow.");
        Int_knob(f,   &haze_blur_passes_, "haze_blur_passes", "Haze Blur Passes");
        Tooltip(f, "Number of box-blur passes for the haze. 3 approximates a Gaussian.");

        Divider(f, "Starburst");
        Float_knob(f, &starburst_gain_,  "starburst_gain",  "Starburst Gain");
        Tooltip(f, "Diffraction spike intensity multiplier. 0 = off.\n\n"
                   "The starburst is computed from the Fraunhofer diffraction pattern of "
                   "the aperture (squared magnitude of the 2D FFT of the aperture mask). "
                   "This is the physically correct origin of diffraction spikes.\n\n"
                   "Spikes are only visible with a polygonal aperture (Aperture Blades >= 3). "
                   "A circular aperture (0 blades) produces Airy rings instead.\n\n"
                   "The pattern is wavelength-dependent: blue light diffracts less than red, "
                   "producing a blue centre and red tips along each spike.\n\n"
                   "Typical values: 50-500.");
        Float_knob(f, &starburst_scale_, "starburst_scale", "Starburst Scale");
        Tooltip(f, "Starburst spike length as a fraction of the image diagonal. "
                   "Larger values produce longer spikes. Default 0.15.");

        Divider(f, "Diagnostics");
        Bool_knob(f, &show_sources_, "show_sources", "Show Sources");
        Tooltip(f, "Output detected bright sources in the source.rgb channels "
                   "for threshold and downsample tuning.");

        Divider(f, "Output");
        static const char* const kOutputModes[] = {
            "Separate channels", "Flare as RGB", "Sources as RGB", nullptr
        };
        Enumeration_knob(f, &output_mode_, kOutputModes, "output_mode", "Output Mode");
        Tooltip(f, "Separate channels: flare data written only to flare.rgb and source.rgb.\n"
                   "Flare as RGB: flare also written to the main RGB channels (input RGB discarded).\n"
                   "Sources as RGB: source map also written to the main RGB channels.");

        // ---- Per-pair toggle tab ----
        Tab_knob(f, "Pairs");
        Button(f, "pairs_refresh",      "Refresh Pairs");
        Button(f, "pairs_select_all",   "Select All");
        Button(f, "pairs_deselect_all", "Deselect All");
        for (int k = 0; k < MAX_PAIRS_UI; ++k)
        {
            Bool_knob(f, &pair_enabled_[k], s_pair_knob_names.names[k], pair_labels_[k]);
            // Force each checkbox onto its own line and hide until a lens loads.
            SetFlags(f, Knob::STARTLINE | Knob::HIDDEN);
            Tooltip(f, "Enable or disable this ghost reflection pair. "
                       "Pairs are shown after a lens file is loaded.");
        }
    }

    // ---- knob_changed ----
    int knob_changed(Knob* k) override
    {
        if (k->is("pairs_refresh")) {
            rebuild_pair_ui();
            return 1;
        }
        if (k->is("pairs_select_all") || k->is("pairs_deselect_all")) {
            const bool val = k->is("pairs_select_all");
            const int n = std::min((int)all_lens_pairs_.size(), MAX_PAIRS_UI);
            for (int i = 0; i < n; ++i) {
                pair_enabled_[i] = val;
                Knob* kn = knob(s_pair_knob_names.names[i]);
                if (kn) kn->set_value(val ? 1.0 : 0.0);
            }
            return 1;
        }
        if (k->is("fov_use_sensor") || k->is("fov_auto_v")) {
            const bool direct = !fov_use_sensor_;
            knob("fov_h")->enable(direct);
            knob("fov_auto_v")->enable(direct);
            knob("fov_v")->enable(direct && !fov_auto_v_);
            knob("sensor_preset")->enable(fov_use_sensor_);
            const bool custom = (sensor_preset_ == 0);
            knob("sensor_w")->enable(fov_use_sensor_ && custom);
            knob("sensor_h")->enable(fov_use_sensor_ && custom);
            knob("focal_length")->enable(fov_use_sensor_);
            return 1;
        }
        if (k->is("sensor_preset")) {
            if (sensor_preset_ > 0) {
                const SensorPreset& p = kSensorPresets[sensor_preset_];
                sensor_w_mm_ = p.w;
                sensor_h_mm_ = p.h;
                knob("sensor_w")->set_value(sensor_w_mm_);
                knob("sensor_h")->set_value(sensor_h_mm_);
            }
            const bool custom = (sensor_preset_ == 0);
            knob("sensor_w")->enable(custom);
            knob("sensor_h")->enable(custom);
            return 1;
        }
        if (k->is("lens_file")) {
            // Reload lens interactively and rebuild pair UI immediately.
            const std::string path(lens_file_ ? lens_file_ : "");
            if (!path.empty() && path != last_lens_file_) {
                if (lens_.load(path.c_str())) {
                    printf("FlareSim: loaded lens '%s' (%d surfaces)\n",
                           lens_.name.c_str(), lens_.num_surfaces());
                    last_lens_file_ = path;
                    rebuild_pair_ui();  // also populates all_lens_pairs_
                } else {
                    fprintf(stderr, "FlareSim: failed to load lens: %s\n", path.c_str());
                }
            }
            return 1;
        }
        return Iop::knob_changed(k);
    }

    // ---- _validate ----
    void _validate(bool for_real) override
    {
        copy_info();

        info_.turn_on(kFlareRed);
        info_.turn_on(kFlareGreen);
        info_.turn_on(kFlareBlue);
        info_.turn_on(kSourceRed);
        info_.turn_on(kSourceGreen);
        info_.turn_on(kSourceBlue);
        info_.turn_on(kHazeRed);
        info_.turn_on(kHazeGreen);
        info_.turn_on(kHazeBlue);
        info_.turn_on(kStarburstRed);
        info_.turn_on(kStarburstGreen);
        info_.turn_on(kStarburstBlue);

        if (!for_real) return;

        input0().validate(for_real);
        if (input(1)) input(1)->validate(for_real);

        const int x0 = info_.x();
        const int y0 = info_.y();
        const int x1 = info_.r();
        const int y1 = info_.t();
        const int w  = x1 - x0;
        const int h  = y1 - y0;
        if (w <= 0 || h <= 0) return;

        // Load lens file (only when the path changes)
        const std::string cur_lens(lens_file_ ? lens_file_ : "");
        if (cur_lens != last_lens_file_)
        {
            if (!cur_lens.empty())
            {
                if (!lens_.load(cur_lens.c_str()))
                    fprintf(stderr, "FlareSim: failed to load lens file: %s\n",
                            cur_lens.c_str());
                else
                    printf("FlareSim: loaded lens '%s' (%d surfaces)\n",
                           lens_.name.c_str(), lens_.num_surfaces());
            }
            last_lens_file_ = cur_lens;
            rebuild_pair_ui();  // also populates all_lens_pairs_
        }

        {
            std::lock_guard<std::mutex> lock(compute_mutex_);
            if (!pending_cuda_error_.empty()) {
                std::string msg;
                msg.swap(pending_cuda_error_);
                error("%s", msg.c_str());
                return;
            }
            pending_x0_       = x0;
            pending_y0_       = y0;
            pending_x1_       = x1;
            pending_y1_       = y1;
            pending_w_        = w;
            pending_h_        = h;
            const Format& fmt = format();
            pending_fmt_x0_   = fmt.x();
            pending_fmt_y0_   = fmt.y();
            pending_fmt_w_    = fmt.width();
            pending_fmt_h_    = fmt.height();
            pending_all_pairs_ = all_lens_pairs_;  // thread-safe snapshot
            needs_compute_ = true;
        }
    }

    // ---- do_compute ----
    void do_compute()
    {
        printf("FlareSim: --- frame %d ---\n", (int)outputContext().frame());
        fflush(stdout);
        const auto t_start = std::chrono::steady_clock::now();

        const int w = pending_w_;
        const int h = pending_h_;
        const size_t npx = (size_t)w * h;

        std::vector<float> out_r(npx, 0.0f), out_g(npx, 0.0f), out_b(npx, 0.0f);
        std::vector<float> src_r(npx, 0.0f), src_g(npx, 0.0f), src_b(npx, 0.0f);
        std::vector<float> haz_r(npx, 0.0f), haz_g(npx, 0.0f), haz_b(npx, 0.0f);
        std::vector<float> sb_r (npx, 0.0f), sb_g (npx, 0.0f), sb_b (npx, 0.0f);

        auto publish = [&]() {
            ghost_r_     = std::move(out_r);
            ghost_g_     = std::move(out_g);
            ghost_b_     = std::move(out_b);
            source_r_    = std::move(src_r);
            source_g_    = std::move(src_g);
            source_b_    = std::move(src_b);
            haze_r_      = std::move(haz_r);
            haze_g_      = std::move(haz_g);
            haze_b_      = std::move(haz_b);
            starburst_r_ = std::move(sb_r);
            starburst_g_ = std::move(sb_g);
            starburst_b_ = std::move(sb_b);
            cache_width_  = w;
            cache_height_ = h;
        };

        if (lens_.surfaces.empty()) { publish(); return; }

        const int x0 = pending_x0_;
        const int y0 = pending_y0_;
        const int x1 = pending_x1_;
        const int y1 = pending_y1_;

        float fov_h, fov_v;
        if (fov_use_sensor_) {
            const float fl = std::max(focal_length_mm_, 0.1f);
            fov_h = 2.0f * std::atan(sensor_w_mm_ / (2.0f * fl));
            fov_v = 2.0f * std::atan(sensor_h_mm_ / (2.0f * fl));
        } else {
            fov_h = (float)(fov_h_deg_ * M_PI / 180.0);
            fov_v = fov_auto_v_
                ? 2.0f * std::atan(std::tan(fov_h * 0.5f) / ((float)pending_fmt_w_ / (float)pending_fmt_h_))
                : (float)(fov_v_deg_ * M_PI / 180.0);
        }

        const float tan_half_h = std::tan(fov_h * 0.5f);
        const float tan_half_v = std::tan(fov_v * 0.5f);
        const int ds = std::max(downsample_, 1);
        const int dw = std::max(1, w / ds);
        const int dh = std::max(1, h / ds);

        std::vector<float> blk_r(dw * dh, 0.0f);
        std::vector<float> blk_g(dw * dh, 0.0f);
        std::vector<float> blk_b(dw * dh, 0.0f);
        std::vector<float> blk_mask(dw * dh, 0.0f);
        std::vector<int>   blk_cnt(dw * dh, 0);

        const bool has_mask = (input(1) != nullptr);

        Row input_row(x0, x1);
        Row mask_row(x0, x1);
        for (int iy = y0; iy < y1; ++iy)
        {
            input0().get(iy, x0, x1, Mask_RGB, input_row);
            if (has_mask)
                input(1)->get(iy, x0, x1, Mask_Alpha, mask_row);
            if (Op::aborted()) { publish(); return; }

            const int dyi = std::min((iy - y0) / ds, dh - 1);
            const float* rp = input_row[Chan_Red];
            const float* gp = input_row[Chan_Green];
            const float* bp = input_row[Chan_Blue];
            const float* mp = has_mask ? mask_row[Chan_Alpha] : nullptr;

            for (int ix = x0; ix < x1; ++ix)
            {
                const int dxi = std::min((ix - x0) / ds, dw - 1);
                const int idx = dyi * dw + dxi;
                blk_r[idx]    += rp[ix];
                blk_g[idx]    += gp[ix];
                blk_b[idx]    += bp[ix];
                blk_mask[idx] += mp ? mp[ix] : 1.0f;
                blk_cnt[idx]  += 1;
            }
        }

        const float fmt_cx = pending_fmt_x0_ + pending_fmt_w_ * 0.5f;
        const float fmt_cy = pending_fmt_y0_ + pending_fmt_h_ * 0.5f;

        std::vector<BrightPixel> sources;
        sources.reserve(1024);

        for (int dyi = 0; dyi < dh; ++dyi)
        {
            for (int dxi = 0; dxi < dw; ++dxi)
            {
                const int cnt = blk_cnt[dyi * dw + dxi];
                if (cnt == 0) continue;

                float r    = blk_r   [dyi * dw + dxi] / cnt;
                float g    = blk_g   [dyi * dw + dxi] / cnt;
                float b    = blk_b   [dyi * dw + dxi] / cnt;
                float luma = 0.2126f*r + 0.7152f*g + 0.0722f*b;

                // Source intensity cap
                if (source_cap_ > 0.0f && luma > source_cap_) {
                    float scale = source_cap_ / luma;
                    r *= scale; g *= scale; b *= scale;
                    luma = source_cap_;
                }

                // Mask: scale intensity and effective luma by average mask value
                const float mask_avg = blk_mask[dyi * dw + dxi] / cnt;
                luma *= mask_avg;
                r    *= mask_avg;
                g    *= mask_avg;
                b    *= mask_avg;

                if (luma < threshold_) continue;

                const int bx0 = x0 + dxi * ds;
                const int bx1 = std::min(bx0 + ds, x1);
                const int by0 = y0 + dyi * ds;
                const int by1 = std::min(by0 + ds, y1);
                const float cx = (bx0 + bx1) * 0.5f;
                const float cy = (by0 + by1) * 0.5f;

                const float ndc_x = (cx - fmt_cx) / pending_fmt_w_;
                const float ndc_y = (cy - fmt_cy) / pending_fmt_h_;

                BrightPixel bp_out;
                bp_out.angle_x = std::atan(ndc_x * 2.0f * tan_half_h);
                bp_out.angle_y = std::atan(ndc_y * 2.0f * tan_half_v);
                bp_out.r = r;  bp_out.g = g;  bp_out.b = b;
                sources.push_back(bp_out);

                // Haze: fill the entire source block with the source colour.
                // Using block-fill (not point-splat) gives the wide blur
                // enough initial coverage to stay visible after spreading.
                if (haze_gain_ > 0.0f) {
                    const int hbx0 = std::max(0,  bx0 - x0);
                    const int hbx1 = std::min(w,  bx1 - x0);
                    const int hby0 = std::max(0,  by0 - y0);
                    const int hby1 = std::min(h,  by1 - y0);
                    for (int py = hby0; py < hby1; ++py)
                        for (int px = hbx0; px < hbx1; ++px) {
                            const size_t idx = (size_t)py * w + px;
                            haz_r[idx] += r;
                            haz_g[idx] += g;
                            haz_b[idx] += b;
                        }
                }
            }
        }

        // ---- Source preview map ----
        if (show_sources_)
        {
            for (int dyi = 0; dyi < dh; ++dyi)
                for (int dxi = 0; dxi < dw; ++dxi)
                {
                    const int cnt = blk_cnt[dyi * dw + dxi];
                    if (cnt == 0) continue;
                    float r    = blk_r   [dyi * dw + dxi] / cnt;
                    float g    = blk_g   [dyi * dw + dxi] / cnt;
                    float b    = blk_b   [dyi * dw + dxi] / cnt;
                    float luma = 0.2126f*r + 0.7152f*g + 0.0722f*b;
                    // Apply source cap before mask — mirrors the sources loop exactly.
                    if (source_cap_ > 0.0f && luma > source_cap_) {
                        const float scale = source_cap_ / luma;
                        r *= scale; g *= scale; b *= scale;
                        luma = source_cap_;
                    }
                    const float mav = blk_mask[dyi * dw + dxi] / cnt;
                    luma *= mav;
                    if (luma < threshold_) continue;
                    const int bx0 = std::min(dxi * ds, w - 1);
                    const int bx1 = std::min(bx0 + ds, w);
                    const int by0 = std::min(dyi * ds, h - 1);
                    const int by1 = std::min(by0 + ds, h);
                    for (int py = by0; py < by1; ++py)
                        for (int px = bx0; px < bx1; ++px) {
                            size_t idx = (size_t)py * w + px;
                            src_r[idx] = r * mav;
                            src_g[idx] = g * mav;
                            src_b[idx] = b * mav;
                        }
                }
        }

        printf("FlareSim: %zu bright source(s) above threshold %.2f\n",
               sources.size(), threshold_);

        // ---- Haze / veiling glare ----
        // haz_r/g/b already filled with source block colours above.
        // Blur first (spreading the energy), then scale by haze_gain so the
        // knob value has a consistent meaning regardless of blur radius.
        if (haze_gain_ > 0.0f && !sources.empty())
        {
            if (haze_radius_ > 0.0f && haze_blur_passes_ > 0)
            {
                const float diag   = std::sqrt((float)w * w + (float)h * h);
                const int   radius = std::max(1, (int)std::round(haze_radius_ * diag));
                printf("FlareSim: haze blur radius %d px, %d pass(es)\n",
                       radius, haze_blur_passes_);
                std::vector<float> blur_tmp;
                box_blur(haz_r.data(), w, h, radius, haze_blur_passes_, blur_tmp);
                box_blur(haz_g.data(), w, h, radius, haze_blur_passes_, blur_tmp);
                box_blur(haz_b.data(), w, h, radius, haze_blur_passes_, blur_tmp);
            }
            // Apply gain after blur so the parameter scales brightness directly.
            const size_t npx_h = (size_t)w * h;
            for (size_t i = 0; i < npx_h; ++i) {
                haz_r[i] *= haze_gain_;
                haz_g[i] *= haze_gain_;
                haz_b[i] *= haze_gain_;
            }
        }

        // ---- Source clustering ----
        // Merge sources that are within cluster_radius_ pixels of each other
        // into a single stronger source.  Runs after haze (which uses the
        // raw per-block positions) but before the GPU launch and max_sources cap.
        if (cluster_radius_ > 0 && (int)sources.size() > 1) {
            const int before = (int)sources.size();
            cluster_sources(sources, cluster_radius_, pending_fmt_w_, tan_half_h);
            if ((int)sources.size() < before)
                printf("FlareSim: clustering: %d -> %d source(s) (radius %d px)\n",
                       before, (int)sources.size(), cluster_radius_);
        }

        last_src_count_.store((int)sources.size());

        if (sources.empty()) { publish(); return; }

        const float sensor_half_w = lens_.focal_length * std::tan(fov_h * 0.5f);
        const float sensor_half_h = lens_.focal_length * std::tan(fov_v * 0.5f);

        GhostConfig cfg;
        cfg.ray_grid              = ray_grid_;
        cfg.gain                  = flare_gain_;
        cfg.aperture_blades       = aperture_blades_;
        cfg.aperture_rotation_deg = aperture_rotation_;
        static const int kSpecCounts[] = {3, 5, 7, 9, 11};
        cfg.spectral_samples      = kSpecCounts[std::max(0, std::min(spectral_idx_, 4))];
        cfg.pupil_jitter          = pupil_jitter_;
        cfg.pupil_jitter_seed     = jitter_auto_seed_
                                      ? (int)outputContext().frame()
                                      : jitter_seed_;

        std::vector<GhostPair> active_pairs;
        std::vector<float>     area_boosts;
        filter_ghost_pairs(lens_, sensor_half_w, sensor_half_h, cfg,
                           active_pairs, area_boosts);

        // Apply per-pair user toggles: keep only pairs the user has enabled.
        {
            std::vector<GhostPair> enabled_pairs;
            std::vector<float>     enabled_boosts;
            enabled_pairs.reserve(active_pairs.size());
            enabled_boosts.reserve(active_pairs.size());

            for (int i = 0; i < (int)active_pairs.size(); ++i)
            {
                int idx = -1;
                for (int k = 0; k < (int)pending_all_pairs_.size(); ++k)
                    if (pending_all_pairs_[k].surf_a == active_pairs[i].surf_a &&
                        pending_all_pairs_[k].surf_b == active_pairs[i].surf_b) {
                        idx = k; break;
                    }
                const bool enabled = (idx < 0 || idx >= MAX_PAIRS_UI) || pair_enabled_[idx];
                if (enabled) {
                    enabled_pairs.push_back(active_pairs[i]);
                    enabled_boosts.push_back(area_boosts[i]);
                }
            }
            active_pairs = std::move(enabled_pairs);
            area_boosts  = std::move(enabled_boosts);
        }

        last_pair_count_.store((int)active_pairs.size());
        printf("FlareSim: %zu active ghost pair(s)\n", active_pairs.size());

        if (active_pairs.empty()) { publish(); return; }

        // Apply user-configurable source count limit (max_sources knob).
        // We sort by descending luma and keep the N brightest so the most
        // visually significant sources always survive the cut.
        if (max_sources_ > 0 && (int)sources.size() > max_sources_) {
            std::partial_sort(sources.begin(),
                              sources.begin() + max_sources_,
                              sources.end(),
                              [](const BrightPixel& a, const BrightPixel& b) {
                                  float la = 0.2126f*a.r + 0.7152f*a.g + 0.0722f*a.b;
                                  float lb = 0.2126f*b.r + 0.7152f*b.g + 0.0722f*b.b;
                                  return la > lb;
                              });
            sources.resize(max_sources_);
            printf("FlareSim: source count capped to %d by Max Sources knob\n",
                   max_sources_);
        }

        pending_cuda_error_.clear();
        const int fmt_x0_in_buf = pending_fmt_x0_ - x0;
        const int fmt_y0_in_buf = pending_fmt_y0_ - y0;

        launch_ghost_cuda(lens_, active_pairs, area_boosts, sources,
                          sensor_half_w, sensor_half_h,
                          out_r.data(), out_g.data(), out_b.data(),
                          w, h,
                          pending_fmt_w_, pending_fmt_h_,
                          fmt_x0_in_buf, fmt_y0_in_buf,
                          cfg, gpu_cache_, &pending_cuda_error_);

        if (ghost_blur_ > 0.0f && ghost_blur_passes_ > 0)
        {
            float diag   = std::sqrt((float)w * w + (float)h * h);
            int   radius = std::max(1, (int)std::round(ghost_blur_ * diag));
            printf("FlareSim: ghost blur radius %d px, %d pass(es)\n",
                   radius, ghost_blur_passes_);
            std::vector<float> blur_tmp;
            box_blur(out_r.data(), w, h, radius, ghost_blur_passes_, blur_tmp);
            box_blur(out_g.data(), w, h, radius, ghost_blur_passes_, blur_tmp);
            box_blur(out_b.data(), w, h, radius, ghost_blur_passes_, blur_tmp);
        }

        // ---- Starburst / diffraction spikes ----
        // Physically based: squared magnitude of the 2D FFT of the aperture mask.
        // Chromatic: R/G/B channels are sampled at different scales (λ/550 nm).
        if (starburst_gain_ > 0.0f && !sources.empty())
        {
            // Recompute PSF only when the aperture shape changes.
            if (starburst_psf_.empty()
                || aperture_blades_   != last_sb_blades_
                || aperture_rotation_ != last_sb_rotation_)
            {
                StarburstConfig sb_probe;
                sb_probe.aperture_blades       = aperture_blades_;
                sb_probe.aperture_rotation_deg = aperture_rotation_;
                compute_starburst_psf(sb_probe, starburst_psf_);
                last_sb_blades_   = aperture_blades_;
                last_sb_rotation_ = aperture_rotation_;
                printf("FlareSim: computed starburst PSF (%dx%d, %d blades)\n",
                       starburst_psf_.N, starburst_psf_.N, aperture_blades_);
            }

            StarburstConfig sb_cfg;
            sb_cfg.gain                  = starburst_gain_;
            sb_cfg.scale                 = starburst_scale_;
            sb_cfg.aperture_blades       = aperture_blades_;
            sb_cfg.aperture_rotation_deg = aperture_rotation_;

            const float th = std::tan(fov_h * 0.5f);
            const float tv = std::tan(fov_v * 0.5f);

            render_starburst(starburst_psf_, sb_cfg, sources,
                             th, tv,
                             sb_r.data(), sb_g.data(), sb_b.data(),
                             w, h,
                             pending_fmt_w_, pending_fmt_h_,
                             fmt_x0_in_buf, fmt_y0_in_buf);
        }

        const auto t_end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        printf("FlareSim: render time %.2f s\n", elapsed);
        fflush(stdout);

        publish();
    }

    // ---- engine ----
    void engine(int y, int x, int r,
                ChannelMask channels, Row& row) override
    {
        {
            std::lock_guard<std::mutex> lock(compute_mutex_);
            if (needs_compute_) {
                needs_compute_ = false;
                do_compute();
            }
        }

        // Pass non-managed channels through from input.
        {
            ChannelSet pass_through(channels);
            pass_through -= kFlareRed;   pass_through -= kFlareGreen;  pass_through -= kFlareBlue;
            pass_through -= kSourceRed;  pass_through -= kSourceGreen; pass_through -= kSourceBlue;
            pass_through -= kHazeRed;       pass_through -= kHazeGreen;      pass_through -= kHazeBlue;
            pass_through -= kStarburstRed;  pass_through -= kStarburstGreen; pass_through -= kStarburstBlue;
            if (output_mode_ != 0) {
                pass_through -= Chan_Red;
                pass_through -= Chan_Green;
                pass_through -= Chan_Blue;
            }
            if (pass_through)
                input0().get(y, x, r, pass_through, row);
        }

        const int  cache_y     = y - info_.y();
        const bool valid_cache = !ghost_r_.empty() && cache_y >= 0 && cache_y < cache_height_;
        const size_t row_off   = valid_cache ? (size_t)cache_y * cache_width_ : 0;
        const int    x_off     = info_.x();

        auto write_buf = [&](Channel ch, const std::vector<float>& buf) {
            if (!channels.contains(ch)) return;
            float* dst = row.writable(ch);
            for (int i = x; i < r; ++i) {
                const int xi = i - x_off;
                dst[i] = (valid_cache && !buf.empty() && xi >= 0 && xi < cache_width_)
                         ? buf[row_off + xi] : 0.0f;
            }
        };

        write_buf(kFlareRed,    ghost_r_);
        write_buf(kFlareGreen,  ghost_g_);
        write_buf(kFlareBlue,   ghost_b_);
        write_buf(kSourceRed,   source_r_);
        write_buf(kSourceGreen, source_g_);
        write_buf(kSourceBlue,  source_b_);
        write_buf(kHazeRed,        haze_r_);
        write_buf(kHazeGreen,      haze_g_);
        write_buf(kHazeBlue,       haze_b_);
        write_buf(kStarburstRed,   starburst_r_);
        write_buf(kStarburstGreen, starburst_g_);
        write_buf(kStarburstBlue,  starburst_b_);

        if (output_mode_ == 1) {
            write_buf(Chan_Red,   ghost_r_);
            write_buf(Chan_Green, ghost_g_);
            write_buf(Chan_Blue,  ghost_b_);
        } else if (output_mode_ == 2) {
            write_buf(Chan_Red,   source_r_);
            write_buf(Chan_Green, source_g_);
            write_buf(Chan_Blue,  source_b_);
        }
    }

    // ---- _request ----
    void _request(int x, int y, int r, int t,
                  ChannelMask channels, int count) override
    {
        // Always request the full frame from input 0 (RGB) for source detection.
        input0().request(info_.x(), info_.y(), info_.r(), info_.t(),
                         Mask_RGB, count);

        // If a mask is connected, request its alpha over the full frame too.
        if (input(1))
            input(1)->request(info_.x(), info_.y(), info_.r(), info_.t(),
                              Mask_Alpha, count);

        // Pass non-managed channels through for the requested region only.
        ChannelSet passthru(channels);
        passthru -= Mask_RGB;
        passthru -= kFlareRed;   passthru -= kFlareGreen;  passthru -= kFlareBlue;
        passthru -= kSourceRed;  passthru -= kSourceGreen; passthru -= kSourceBlue;
        passthru -= kHazeRed;       passthru -= kHazeGreen;      passthru -= kHazeBlue;
        passthru -= kStarburstRed;  passthru -= kStarburstGreen; passthru -= kStarburstBlue;
        if (passthru)
            input0().request(x, y, r, t, passthru, count);
    }
};

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

static Iop* build(Node* node) { return new FlareSim(node); }

const Iop::Description FlareSim::d(
    "FlareSim",
    "Filter/FlareSim",
    build);

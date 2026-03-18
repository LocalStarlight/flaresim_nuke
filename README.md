# FlareSim

A CUDA-accelerated Nuke plugin for physically-based lens flare simulation.
FlareSim traces rays through a real lens prescription to produce ghost
reflections that respond correctly to source position, wavelength, and aperture
shape — without look-up textures or artist-painted elements.

---

## Credits

FlareSim is built on the foundational work of **Eamonn Nugent**
([@space55](https://github.com/space55) · [55.dev](https://55.dev/)), whose
original CPU-based lens flare renderer established the core ray-tracing
approach, optical physics, and lens file format that this project builds upon.
Without that work, FlareSim would not exist.
The original codebase can be found at
[github.com/space55/blackhole-rt](https://github.com/space55/blackhole-rt/).

---

## How it works

FlareSim takes any image as input and treats bright pixels as light sources.
For each source it fires a grid of rays across the entrance pupil of the lens,
tracing them through every optical surface. At two user-selected surfaces the
ray reflects (instead of transmitting) — this is a ghost bounce pair. The
reflected ray continues to the sensor plane, where its contribution is splatted
onto the output buffer.

Because the tracing is done on the actual surface geometry, the resulting
ghosts automatically exhibit:

- **Correct position** — ghosts land exactly where the optics dictate, including
  the characteristic axis-flipping behaviour when the source crosses the frame
  centre
- **Chromatic aberration** — each ghost is traced at multiple wavelengths;
  surfaces with AR coatings and dispersive glass bend R/G/B by different amounts
- **Aperture shape** — the ghost footprint inherits the polygonal or circular
  aperture set on the node
- **Fresnel weighting** — each bounce is weighted by the reflectance of its
  surface, including single-layer MgF2 AR coating simulation

The kernel runs entirely on GPU (NVIDIA CUDA). Render times are typically
under one second per frame for standard HD/2K images.

---

## Installation

1. Copy `FlareSim.dll` to a directory on your `NUKE_PATH`,
   for example `%USERPROFILE%/.nuke/plugins/`.
2. Copy `menu.py` to the same directory.
   If you already have a `menu.py` there, add this line to it instead:
   ```python
   nuke.menu('Nodes').addCommand('Filter/FlareSim', 'nuke.createNode("FlareSim")')
   ```
3. Copy the `lenses/` folder somewhere accessible and point the **Lens File**
   knob at one of the `.lens` files.
4. Restart Nuke. `FlareSim` will appear in the node menu under **Filter**.

**Requirements:** Windows, Nuke 16, an NVIDIA GPU (any card from 2012 onward,
compute capability 3.0+), CUDA runtime 13.

---

## Quick start

1. Connect your beauty plate (or a sky/exterior shot with visible light sources)
   to FlareSim's input.
2. Point **Lens File** at a `.lens` prescription — `doublegauss.lens` is a good
   starting point.
3. Set **FOV H** to match the horizontal field of view of your camera.
4. Increase **Threshold** until only the brightest highlights register as sources
   (watch the `source.rgb` channels, or enable **Show Sources**).
5. Adjust **Flare Gain** to taste.
6. Use the **Pairs** tab to disable any ghost pairs that are unwanted.

---

## Output channels

FlareSim writes several named channel groups in addition to passing the input
RGB through:

| Channel | Contents |
|---------|----------|
| `flare.rgb` | Ghost reflections only |
| `source.rgb` | Map of detected bright sources (useful for threshold tuning) |
| `haze.rgb` | Veiling glare / scattered light glow |
| `starburst.rgb` | Diffraction spikes |

What appears in the main `rgba` channels depends on the **Output Mode** setting.

---

## Parameters

### Ghost

**Flare Gain**
Overall intensity multiplier for the ghost reflections. Because ghost Fresnel
weights are very small (two reflections, each typically 1–4%), a gain of
several hundred is normal.

**Ray Grid (NxN)**
Number of entrance-pupil samples per dimension. The kernel fires N² rays per
(source, ghost pair) combination. Higher values produce smoother, less noisy
ghosts at the cost of render time. 32–64 is a good starting range; 128 is
high quality.

**Pupil Jitter**
Sampling pattern for the entrance-pupil grid:
- *Off* — regular NxN grid. Fastest; can show a faint dot pattern at low ray
  counts.
- *Stratified* — one randomly-placed sample per grid cell. Breaks up the dot
  pattern. Use with **Auto Seed** for automatic per-frame variation.
- *Halton* — quasi-random low-discrepancy sequence. Deterministic (no seed
  needed), very even coverage. Good for still frames and motion-blur renders.

**Jitter Seed**
Integer seed for the Stratified noise pattern. Only used when **Auto Seed** is
off. Has no effect in Off or Halton modes.

**Auto Seed**
When on (the default), the seed is automatically set to the current frame
number. Sample positions change on every frame with no manual keyframing
required — equivalent to animating Jitter Seed = frame. Turn off to use the
fixed Jitter Seed value, which gives a repeatable pattern useful for still
renders or multi-pass workflows where the noise must match exactly.

**Threshold**
Minimum luminance (in scene-linear values) for a pixel to register as a light
source. Raise this to suppress dim sources and reduce render time. Use
**Show Sources** to see exactly what is being detected.

**Source Cap**
Clamps the luminance of any single source before it contributes to the flare.
Useful when one extremely bright highlight would otherwise dominate all ghosts.
`0` = off.

**Max Sources**
Maximum number of bright sources passed to the GPU per frame. When there are
more sources than this limit, the N brightest (by luma) are kept and the rest
are discarded. `0` = unlimited.

Reducing this is the primary tool for managing render time on frames with many
sources (e.g. sparkly water, city nights). It also prevents runaway render
times if the threshold is accidentally set too low.

**Downsample**
Stride used when scanning the input image for bright sources. A value of 4
samples every 4th pixel in each axis (1/16th of all pixels). Lower values find
more sources and are more accurate but slower. For most shots 4–8 is sufficient.

---

### Camera

These settings tell FlareSim the field of view of the camera, which it uses to
convert pixel positions into ray angles.

**Use Sensor Size**
When enabled, FOV is computed from **Sensor Width**, **Sensor Height**, and
**Focal Length** — matching the way most real-world cameras are specified.
When disabled, enter the FOV angles directly.

**Sensor Preset**
Fills **Sensor Width** and **Sensor Height** from a list of common formats
(Super35, Full Frame, Micro Four Thirds, etc.).

**FOV H / FOV V (deg)**
Horizontal and vertical field of view in degrees. Only active when
**Use Sensor Size** is off.

**Auto FOV V**
Derives vertical FOV from horizontal FOV and the image aspect ratio. Disable
this only if your image has non-square pixels or a non-standard crop.

**Sensor Width / Height (mm)** and **Focal Length (mm)**
Physical sensor dimensions and focal length. Only active when
**Use Sensor Size** is on.

---

### Aperture

**Aperture Blades**
`0` = circular aperture. `3`–`16` = regular polygon with that many sides.
The aperture shape directly affects the footprint of each ghost: a hexagonal
aperture produces hexagonal ghosts, matching what a real 6-bladed iris would
produce.

**Aperture Rotation**
Rotates the polygonal aperture in degrees. Useful to match a specific iris
orientation captured in footage.

---

### Spectral

**Spectral Samples**
Number of wavelengths sampled per ray. More samples produce smoother colour
fringing at the cost of render time.

| Setting | Wavelengths | Use |
|---------|-------------|-----|
| 3 (R/G/B) | 650 / 550 / 450 nm | Fast; one sample per channel |
| 5 | 400–700 nm, 5 steps | Good balance |
| 7–11 | Finer steps | High quality; smooth chromatic aberration |

---

### Post-process

**Ghost Blur / Ghost Blur Passes**
Applies a fast box blur to the ghost buffer after rendering. Radius is
expressed as a fraction of the image diagonal (e.g. `0.003` on a 1920-wide
image = ~7 pixels). Multiple passes approximate a Gaussian; 3 passes is
usually sufficient.

This is useful for adding a small amount of out-of-focus softness to the
ghosts, or for hiding the point-splat nature of the renderer at low ray counts.

**Haze Gain**
Enables veiling glare — a broad, soft glow written to `haze.rgb`. Each source
paints its colour into a block of the haze buffer, which is then blurred with
a wide radius. This simulates light scattered inside the lens barrel.
`0` = off.

**Haze Radius / Haze Blur Passes**
Blur radius (as a fraction of the image diagonal) and number of passes for the
haze glow. Larger radii produce a more diffuse, "milky" effect.

---

### Starburst

**Starburst Gain**
Enables diffraction spikes written to `starburst.rgb`. `0` = off.

The starburst is computed from the squared magnitude of the 2D FFT of the
aperture mask — the physically correct Fraunhofer diffraction pattern. Spike
count, angle, and falloff all emerge from the aperture geometry: a hexagonal
aperture produces 6 spikes, an octagonal aperture 8 spikes, and so on. A
circular aperture (`0` blades) produces Airy rings rather than spikes.

The pattern is wavelength-dependent — red light diffracts more than blue,
so the spikes have a blue centre and red tips. This colouration is computed
per-channel (R/G/B at 650/550/450 nm) and requires no artistic tuning.

Typical gain values are 50–500.

**Starburst Scale**
Controls how far the spikes extend, expressed as a fraction of the image
diagonal. `0.15` means spikes reach roughly 15% of the diagonal from the
source — about 330 pixels on a 1920×1080 image.

---

### Diagnostics

**Show Sources**
Writes the detected source map into the `source.rgb` channels regardless of
the current Output Mode. Use this to tune **Threshold** and **Downsample** —
you should see the highlights you want to flare, and nothing else.

---

### Output

**Output Mode**
Controls which data appears in the main `rgba` channels:

- *Separate channels* — input RGB passes through unchanged; flare data is only
  in `flare.rgb` / `source.rgb` / `haze.rgb`. Use this when compositing
  manually downstream.
- *Flare as RGB* — the ghost contribution is written into the main RGB as well
  as `flare.rgb`. The original input RGB is discarded.
- *Sources as RGB* — the source map is written into main RGB. Useful for
  threshold inspection without needing to switch viewer channels.

---

## Pairs tab

The Pairs tab lists every ghost bounce pair that the loaded lens produces, one
checkbox per pair. A pair is labelled `(A, B)` where A and B are the two
surface indices (0-based) at which the ray reflects.

FlareSim pre-filters the list and only shows pairs whose estimated Fresnel
reflectance is above a minimum threshold, so very dim pairs are already hidden.

**Refresh Pairs** rebuilds the list from the current lens file and FOV settings.
Use this if you change the lens or FOV and the pair list looks stale.

**Select All / Deselect All** toggle all pairs at once.

Disabling a pair completely removes it from the GPU workload, so unchecking
pairs that don't contribute visually is a free render-time saving.

---

## Lens files

Lens files are plain-text `.lens` prescriptions. Each file describes one lens
as an ordered list of optical surfaces from front (entrance) to back (sensor).

```
name: Double Gauss 58mm f/2
focal_length: 58.0

surfaces:
# radius    thickness   ior     abbe    semi_ap  coating
  39.68      8.36       1.670   47.2    25.0     1
  152.73     0.20       1.000   0.0     25.0     0
  31.58      7.38       1.670   47.2    21.0     1
  0          4.10       1.699   30.1    21.0     1
  -1000.0   16.50       1.000   0.0     14.0     0
  stop       12.13      1.000   0.0     12.0     0
  ...
```

| Column | Meaning |
|--------|---------|
| `radius` | Signed radius of curvature in mm. `0` = flat surface. |
| `thickness` | Axial distance to the next surface in mm. |
| `ior` | Refractive index of the medium **after** this surface (d-line, 587.6 nm). Use `1.0` for air. |
| `abbe` | Abbe V-number for dispersion. `0` = non-dispersive (air/vacuum). |
| `semi_ap` | Clear semi-diameter (half the clear aperture) in mm. |
| `coating` | AR coating layers. `0` = bare glass. `1` = single-layer MgF₂ quarter-wave. Higher values further reduce reflectance. |

The special keyword `stop` in the radius column marks the aperture stop surface.

Lines beginning with `#` are comments. The `focal_length` field is used to
convert FOV angles to physical sensor dimensions; set it to the nominal focal
length of the lens.

A large library of real lens prescriptions is included in `lenses/lens_files/`,
covering Nikon primes and zooms, Canon lenses, ARRI/Zeiss cinema lenses,
Cooke triplets, mirror lenses, and more.

---

## Performance tips

- **Ray Grid** is the biggest render-time lever. Start at 32 and raise to 64
  or 128 only where needed.
- **Threshold** is the second biggest. Every source above threshold costs
  (n_pairs × n_grid²) ray traces. Fewer sources = much faster renders.
- **Max Sources** sets a hard ceiling on GPU work per frame, preventing
  surprise-slow frames on shots with many highlights.
- **Disable pairs** in the Pairs tab that don't contribute visually — they cost
  nothing when unchecked.
- Lenses with more elements produce more ghost pairs (an N-surface lens has
  N(N-1)/2 possible pairs). The doublegauss (11 surfaces) has 55; a complex
  zoom may have several hundred.

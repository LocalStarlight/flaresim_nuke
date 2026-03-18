// ============================================================================
// trace.h — Ray tracing through a sequential lens system
// ============================================================================
#pragma once

#include "vec3.h"
#include "lens.h"

// Result of tracing a ray through the lens to the sensor plane.
struct TraceResult
{
    Vec3f position; // landing position on sensor (mm)
    float weight;   // total Fresnel weight (reflectance at bounces × transmittance elsewhere)
    bool valid;     // did the ray reach the sensor without vignetting or TIR?
};

// Intersect a ray with a single lens surface (sphere or flat).
// Returns false if the ray misses the surface or its clear aperture.
// On success, sets hit_pos and outward normal (opposing ray direction).
bool intersect_surface(const Ray &ray, const Surface &surf,
                       Vec3f &hit_pos, Vec3f &normal);

// Trace a ghost ray through the complete lens system.
//
// The ray enters the front of the lens and transmits through all surfaces
// except at bounce_a and bounce_b (where it reflects).  The path is:
//
//   Phase 1: forward through surfaces 0..bounce_b  (reflect at bounce_b)
//   Phase 2: backward through surfaces bounce_b-1..bounce_a  (reflect at bounce_a)
//   Phase 3: forward through surfaces bounce_a+1..N-1 to the sensor
//
// lambda_nm is used for dispersion (wavelength-dependent IOR) and coating
// reflectance calculations.
//
// Requires: 0 <= bounce_a < bounce_b < lens.num_surfaces()
TraceResult trace_ghost_ray(const Ray &ray, const LensSystem &lens,
                            int bounce_a, int bounce_b,
                            float lambda_nm);

// ============================================================================
// lens.h — Lens system data structures and file parser
// ============================================================================
#pragma once

#include <string>
#include <vector>
#include "fresnel.h"

// One optical surface in the lens system.
struct Surface
{
    float radius;        // signed radius of curvature in mm (0 = flat)
    float thickness;     // axial distance to next surface (mm)
    float ior;           // refractive index of medium AFTER this surface (d-line)
    float abbe_v;        // Abbe number (0 = air / non-dispersive)
    float semi_aperture; // clear semi-diameter (mm)
    int coating;         // AR coating layers (0 = uncoated)
    bool is_stop;        // is this the aperture stop?

    // Computed by LensSystem::compute_geometry()
    float z; // axial position of surface vertex (mm)

    // Wavelength-dependent IOR (Cauchy via Abbe number)
    float ior_at(float lambda_nm) const
    {
        return dispersion_ior(ior, abbe_v, lambda_nm);
    }
};

// Complete lens system: ordered sequence of surfaces + sensor plane.
struct LensSystem
{
    std::string name;
    float focal_length = 0; // nominal focal length (mm), from file

    std::vector<Surface> surfaces;
    float sensor_z = 0; // axial position of sensor plane (mm), computed

    // Load a lens prescription from a .lens file.
    bool load(const char *filename);

    // Compute surface z positions and sensor_z from thicknesses.
    void compute_geometry();

    // IOR of the medium BEFORE surface idx (air for the first surface).
    float ior_before(int idx) const
    {
        return (idx <= 0) ? 1.0f : surfaces[idx - 1].ior;
    }

    // Wavelength-dependent version.
    float ior_before(int idx, float lambda_nm) const
    {
        if (idx <= 0)
            return 1.0f;
        return surfaces[idx - 1].ior_at(lambda_nm);
    }

    int num_surfaces() const { return (int)surfaces.size(); }

    void print_summary() const;
};

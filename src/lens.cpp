// ============================================================================
// lens.cpp — Lens file parser and geometry computation
// ============================================================================

#include "lens.h"

#include <cstdio>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// .lens file format:
//
//   # comment
//   name: My Lens 50mm f/1.4
//   focal_length: 50.0
//
//   surfaces:
//   # radius   thickness   ior    abbe   semi_ap   coating
//     58.95    7.52        1.670  47.2   25.0      1
//     169.66   0.24        1.000  0.0    25.0      0
//     stop     2.00        1.000  0.0    12.5      0
//     ...
//
//   The last surface's thickness is the back focal distance to the sensor.
//   A radius of "stop" or "STOP" marks the aperture stop (flat surface).
//   A radius of 0 or "inf" means a flat surface.
// ---------------------------------------------------------------------------

bool LensSystem::load(const char *filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        fprintf(stderr, "ERROR: cannot open lens file: %s\n", filename);
        return false;
    }

    surfaces.clear();
    name.clear();
    focal_length = 0;

    bool in_surfaces = false;
    std::string line;

    while (std::getline(file, line))
    {
        // Strip leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos)
            continue;
        line = line.substr(start);

        // Skip comments
        if (line[0] == '#')
            continue;

        // Check for metadata
        if (!in_surfaces)
        {
            if (line.substr(0, 5) == "name:")
            {
                name = line.substr(5);
                size_t ns = name.find_first_not_of(" \t");
                if (ns != std::string::npos)
                    name = name.substr(ns);
                continue;
            }
            if (line.substr(0, 13) == "focal_length:")
            {
                // Imbue with classic locale so '.' is always the decimal separator.
                std::istringstream fl_ss(line.substr(13));
                fl_ss.imbue(std::locale::classic());
                fl_ss >> focal_length;
                continue;
            }
            if (line.substr(0, 9) == "surfaces:")
            {
                in_surfaces = true;
                continue;
            }
            continue;
        }

        // Parse surface line
        // Expected: radius thickness ior abbe semi_aperture coating
        // Imbue with classic locale so '.' is always the decimal separator.
        std::istringstream iss(line);
        iss.imbue(std::locale::classic());
        std::string radius_str;
        iss >> radius_str;
        if (radius_str.empty() || radius_str[0] == '#')
            continue;

        Surface s{};
        s.is_stop = false;

        if (radius_str == "stop" || radius_str == "STOP")
        {
            s.radius = 0.0f;
            s.is_stop = true;
        }
        else if (radius_str == "inf" || radius_str == "INF")
        {
            s.radius = 0.0f;
        }
        else
        {
            std::istringstream rad_ss(radius_str);
            rad_ss.imbue(std::locale::classic());
            rad_ss >> s.radius;
        }

        iss >> s.thickness >> s.ior >> s.abbe_v >> s.semi_aperture >> s.coating;

        if (s.semi_aperture <= 0)
        {
            fprintf(stderr, "WARNING: surface %zu has semi_aperture <= 0, skipping\n",
                    surfaces.size());
            continue;
        }

        surfaces.push_back(s);
    }

    if (surfaces.empty())
    {
        fprintf(stderr, "ERROR: no surfaces found in lens file: %s\n", filename);
        return false;
    }

    compute_geometry();
    return true;
}

void LensSystem::compute_geometry()
{
    float z = 0;
    for (size_t i = 0; i < surfaces.size(); ++i)
    {
        surfaces[i].z = z;
        z += surfaces[i].thickness;
    }
    sensor_z = z;
}

void LensSystem::print_summary() const
{
    printf("Lens: %s\n", name.c_str());
    printf("  focal_length: %.1f mm\n", focal_length);
    printf("  surfaces: %d\n", num_surfaces());
    printf("  sensor_z: %.2f mm\n", sensor_z);
    printf("  %-4s  %10s  %8s  %6s  %6s  %8s  %4s  %s\n",
           "Idx", "Radius", "Thick", "IOR", "Abbe", "SemiAp", "Coat", "");

    for (int i = 0; i < num_surfaces(); ++i)
    {
        const Surface &s = surfaces[i];
        if (s.radius == 0.0f)
        {
            printf("  %-4d  %10s  %8.3f  %6.3f  %6.1f  %8.2f  %4d  %s\n",
                   i, s.is_stop ? "STOP" : "flat",
                   s.thickness, s.ior, s.abbe_v, s.semi_aperture, s.coating,
                   s.is_stop ? "<-- aperture stop" : "");
        }
        else
        {
            printf("  %-4d  %10.3f  %8.3f  %6.3f  %6.1f  %8.2f  %4d\n",
                   i, s.radius, s.thickness, s.ior, s.abbe_v,
                   s.semi_aperture, s.coating);
        }
    }
}

// ============================================================================
// trace.cpp — Ray-surface intersection and sequential lens trace
// ============================================================================

#include "trace.h"
#include "fresnel.h"

#include <cmath>
#include <cstdio>
#include <algorithm>

// Fast normalize: avoids the branch and second divide in Vec3f::normalized().
static inline Vec3f fast_normalize(const Vec3f &v)
{
    float l2 = v.x * v.x + v.y * v.y + v.z * v.z;
    float inv = 1.0f / std::sqrt(l2); // -ffast-math turns this into rsqrt
    return {v.x * inv, v.y * inv, v.z * inv};
}

// ---------------------------------------------------------------------------
// Intersect a ray with a lens surface.
//
// Flat surfaces (radius == 0): ray–plane intersection at z = surf.z.
// Curved surfaces: ray–sphere intersection.
//   Centre of curvature at C = (0, 0, surf.z + surf.radius).
//   We pick the intersection point closest to the surface vertex.
//
// On success: hit_pos is set, normal opposes the ray direction.
// Returns false on miss, aperture clip, or no positive t.
// ---------------------------------------------------------------------------

bool intersect_surface(const Ray &ray, const Surface &surf,
                       Vec3f &hit_pos, Vec3f &normal)
{
    if (std::abs(surf.radius) < 1e-6f)
    {
        // ---- Flat surface ----
        if (std::abs(ray.dir.z) < 1e-12f)
            return false; // parallel

        float t = (surf.z - ray.origin.z) / ray.dir.z;
        if (t < 1e-6f)
            return false;

        hit_pos = ray.origin + ray.dir * t;

        // Aperture check
        float h2 = hit_pos.x * hit_pos.x + hit_pos.y * hit_pos.y;
        if (h2 > surf.semi_aperture * surf.semi_aperture)
            return false;

        // Normal opposing ray direction
        normal = Vec3f(0, 0, (ray.dir.z > 0) ? -1.0f : 1.0f);
        return true;
    }

    // ---- Spherical surface ----
    float R = surf.radius;
    Vec3f center(0, 0, surf.z + R);
    Vec3f oc = ray.origin - center;

    float a = dot(ray.dir, ray.dir);
    float b = 2.0f * dot(oc, ray.dir);
    float c = dot(oc, oc) - R * R;
    float disc = b * b - 4.0f * a * c;

    if (disc < 0)
        return false;

    float sqrt_disc = std::sqrt(disc);
    float inv_2a = 0.5f / a;
    float t1 = (-b - sqrt_disc) * inv_2a;
    float t2 = (-b + sqrt_disc) * inv_2a;

    // Pick the intersection closest to the surface vertex z.
    // Both t must be positive (ahead of the ray).
    float t;
    if (t1 > 1e-6f && t2 > 1e-6f)
    {
        float z1 = ray.origin.z + t1 * ray.dir.z;
        float z2 = ray.origin.z + t2 * ray.dir.z;
        t = (std::abs(z1 - surf.z) < std::abs(z2 - surf.z)) ? t1 : t2;
    }
    else if (t1 > 1e-6f)
        t = t1;
    else if (t2 > 1e-6f)
        t = t2;
    else
        return false;

    hit_pos = ray.origin + ray.dir * t;

    // Aperture check
    float h2 = hit_pos.x * hit_pos.x + hit_pos.y * hit_pos.y;
    if (h2 > surf.semi_aperture * surf.semi_aperture)
        return false;

    // Normal: from sphere centre toward hit point, then ensure it opposes ray
    normal = (hit_pos - center) / std::abs(R);
    if (dot(normal, ray.dir) > 0)
        normal = -normal;

    return true;
}

// ---------------------------------------------------------------------------
// Refract a ray at a surface.
// n_ratio = n1 / n2.  Normal opposes the incoming ray.
// Returns false on total internal reflection.
// ---------------------------------------------------------------------------

static bool refract_direction(const Vec3f &dir, const Vec3f &normal,
                              float n_ratio, Vec3f &out_dir)
{
    float cos_i = -dot(normal, dir);
    float sin2_t = n_ratio * n_ratio * (1.0f - cos_i * cos_i);

    if (sin2_t >= 1.0f)
        return false; // TIR

    float cos_t = std::sqrt(1.0f - sin2_t);
    out_dir = fast_normalize(dir * n_ratio + normal * (n_ratio * cos_i - cos_t));
    return true;
}

// ---------------------------------------------------------------------------
// Reflect a ray off a surface.  Normal opposes the incoming ray.
// ---------------------------------------------------------------------------

static Vec3f reflect_direction(const Vec3f &dir, const Vec3f &normal)
{
    return fast_normalize(dir - normal * (2.0f * dot(dir, normal)));
}

// ---------------------------------------------------------------------------
// Trace a ghost ray through the full lens system.
//
// Three-phase sequential trace with reflections at bounce_a and bounce_b.
// ---------------------------------------------------------------------------

TraceResult trace_ghost_ray(const Ray &ray_in, const LensSystem &lens,
                            int bounce_a, int bounce_b,
                            float lambda_nm)
{
    Ray ray = ray_in; // local mutable copy
    TraceResult result;
    result.valid = false;
    result.weight = 1.0f;

    int N = lens.num_surfaces();
    float current_ior = 1.0f; // start in air

    // ================================================================
    // Phase 1: forward through surfaces 0 .. bounce_b
    //          transmit at all except bounce_b (reflect)
    // ================================================================
    for (int s = 0; s <= bounce_b; ++s)
    {
        Vec3f hit, norm;
        if (!intersect_surface(ray, lens.surfaces[s], hit, norm))
            return result; // vignetted

        ray.origin = hit;

        float n1 = current_ior;
        float n2 = lens.surfaces[s].ior_at(lambda_nm);

        float cos_i = std::abs(dot(norm, ray.dir));
        float R = surface_reflectance(cos_i, n1, n2,
                                      lens.surfaces[s].coating, lambda_nm);

        if (s == bounce_b)
        {
            // Reflect
            ray.dir = reflect_direction(ray.dir, norm);
            result.weight *= R;
            // current_ior unchanged (still in the medium before this surface)
        }
        else
        {
            // Transmit
            Vec3f new_dir;
            if (!refract_direction(ray.dir, norm, n1 / n2, new_dir))
                return result; // TIR
            ray.dir = new_dir;
            result.weight *= (1.0f - R);
            current_ior = n2;
        }
    }

    // ================================================================
    // Phase 2: backward through surfaces bounce_b-1 .. bounce_a
    //          transmit at all except bounce_a (reflect)
    // ================================================================
    for (int s = bounce_b - 1; s >= bounce_a; --s)
    {
        Vec3f hit, norm;
        if (!intersect_surface(ray, lens.surfaces[s], hit, norm))
            return result;

        ray.origin = hit;

        // Backward through surface s:
        //   n1 = medium on the side the ray is coming from = surfaces[s].ior_at(λ)
        //   n2 = medium on the other side = ior_before(s, λ)
        float n1 = current_ior;
        float n2 = lens.ior_before(s, lambda_nm);

        float cos_i = std::abs(dot(norm, ray.dir));
        float R = surface_reflectance(cos_i, n1, n2,
                                      lens.surfaces[s].coating, lambda_nm);

        if (s == bounce_a)
        {
            // Reflect — ray resumes forward direction
            ray.dir = reflect_direction(ray.dir, norm);
            result.weight *= R;
            // After reflecting at bounce_a, we're in the medium to the RIGHT
            // of bounce_a, which is surfaces[bounce_a].ior_at(λ)
            current_ior = lens.surfaces[bounce_a].ior_at(lambda_nm);
        }
        else
        {
            // Transmit backward
            Vec3f new_dir;
            if (!refract_direction(ray.dir, norm, n1 / n2, new_dir))
                return result;
            ray.dir = new_dir;
            result.weight *= (1.0f - R);
            current_ior = n2;
        }
    }

    // ================================================================
    // Phase 3: forward through surfaces bounce_a+1 .. N-1
    //
    // The ray exits bounce_a going forward and passes through all
    // remaining surfaces (including those between bounce_a and bounce_b
    // for a second time, physically correct for a double-bounce ghost).
    // ================================================================
    for (int s = bounce_a + 1; s < N; ++s)
    {
        Vec3f hit, norm;
        if (!intersect_surface(ray, lens.surfaces[s], hit, norm))
            return result;

        ray.origin = hit;

        float n1 = current_ior;
        float n2 = lens.surfaces[s].ior_at(lambda_nm);

        float cos_i = std::abs(dot(norm, ray.dir));
        float R = surface_reflectance(cos_i, n1, n2,
                                      lens.surfaces[s].coating, lambda_nm);

        Vec3f new_dir;
        if (!refract_direction(ray.dir, norm, n1 / n2, new_dir))
            return result;
        ray.dir = new_dir;
        result.weight *= (1.0f - R);
        current_ior = n2;
    }

    // ================================================================
    // Propagate to sensor plane
    // ================================================================
    if (std::abs(ray.dir.z) < 1e-12f)
        return result;

    float t = (lens.sensor_z - ray.origin.z) / ray.dir.z;
    if (t < 0)
        return result; // sensor is behind the ray

    result.position = ray.origin + ray.dir * t;
    result.valid = true;
    return result;
}

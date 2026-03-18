// ============================================================================
// fresnel.h — Fresnel equations, thin-film AR coatings, glass dispersion
// ============================================================================
#pragma once

#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- Dispersion (Cauchy via Abbe number) --------------------------------

// Compute wavelength-dependent IOR from the d-line IOR and Abbe number.
// Reference wavelengths: F = 486.13 nm, d = 587.56 nm, C = 656.27 nm
inline float dispersion_ior(float n_d, float V_d, float lambda_nm)
{
    if (V_d < 0.1f || n_d <= 1.0001f)
        return n_d; // air or non-dispersive

    constexpr float lF = 486.13f;
    constexpr float lC = 656.27f;
    constexpr float ld = 587.56f;

    // n_F - n_C = (n_d - 1) / V_d
    float dn = (n_d - 1.0f) / V_d;

    // Cauchy: n(λ) = A + B/λ²
    float inv_lF2 = 1.0f / (lF * lF);
    float inv_lC2 = 1.0f / (lC * lC);
    float inv_ld2 = 1.0f / (ld * ld);

    float B = dn / (inv_lF2 - inv_lC2);
    float A = n_d - B * inv_ld2;

    return A + B / (lambda_nm * lambda_nm);
}

// ---- Fresnel equations --------------------------------------------------

// Unpolarized Fresnel reflectance at a dielectric interface.
// cos_i: cosine of incidence angle (positive).
inline float fresnel_reflectance(float cos_i, float n1, float n2)
{
    cos_i = std::abs(cos_i);
    float eta = n1 / n2;
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);

    if (sin2_t >= 1.0f)
        return 1.0f; // total internal reflection

    float cos_t = std::sqrt(1.0f - sin2_t);

    float rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    float rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);

    return 0.5f * (rs * rs + rp * rp);
}

// ---- Thin-film AR coating -----------------------------------------------

// Single-layer coating reflectance (thin-film interference).
// coating_n : coating refractive index (e.g. MgF2 = 1.38)
// d_nm      : physical coating thickness in nm
// lambda_nm : light wavelength in nm
inline float coating_reflectance(float cos_i, float n1, float n2,
                                 float coating_n, float d_nm, float lambda_nm)
{
    // Snell into coating
    float sin2_c = (n1 / coating_n) * (n1 / coating_n) * (1.0f - cos_i * cos_i);
    if (sin2_c >= 1.0f)
        return fresnel_reflectance(cos_i, n1, n2);
    float cos_c = std::sqrt(1.0f - sin2_c);

    // Phase thickness (double pass through the film)
    float delta = 2.0f * (float)M_PI * coating_n * d_nm * cos_c / lambda_nm;

    // Fresnel amplitude coefficients at each interface
    float r01 = (n1 * cos_i - coating_n * cos_c) / (n1 * cos_i + coating_n * cos_c);

    float sin2_2 = (coating_n / n2) * (coating_n / n2) * (1.0f - cos_c * cos_c);
    if (sin2_2 >= 1.0f)
        return fresnel_reflectance(cos_i, n1, n2);
    float cos_2 = std::sqrt(1.0f - sin2_2);
    float r12 = (coating_n * cos_c - n2 * cos_2) / (coating_n * cos_c + n2 * cos_2);

    // Airy formula for total reflectance
    float cos_2delta = std::cos(2.0f * delta);
    float num = r01 * r01 + r12 * r12 + 2.0f * r01 * r12 * cos_2delta;
    float den = 1.0f + r01 * r01 * r12 * r12 + 2.0f * r01 * r12 * cos_2delta;

    return std::clamp(num / den, 0.0f, 1.0f);
}

// ---- Combined surface reflectance ---------------------------------------

// Returns Fresnel reflectance at a lens surface, accounting for AR coating.
// coating_layers: 0 = uncoated, 1 = single-layer MgF2, 2+ ≈ multi-coat
inline float surface_reflectance(float cos_i, float n1, float n2,
                                 int coating_layers, float lambda_nm)
{
    if (coating_layers <= 0)
        return fresnel_reflectance(cos_i, n1, n2);

    // MgF2 single-layer: n=1.38, quarter-wave thickness at 550 nm
    constexpr float mgf2_n = 1.38f;
    constexpr float design_lambda = 550.0f;               // nm
    float qw_thickness = design_lambda / (4.0f * mgf2_n); // ≈ 99.6 nm

    float R = coating_reflectance(cos_i, n1, n2, mgf2_n, qw_thickness, lambda_nm);

    // Multi-layer coatings give progressively lower reflectance
    // (very rough approximation — real multi-layer stacks are more complex)
    for (int i = 1; i < coating_layers; ++i)
        R *= 0.25f;

    return std::clamp(R, 0.0f, 1.0f);
}

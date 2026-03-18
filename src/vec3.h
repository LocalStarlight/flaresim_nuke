// ============================================================================
// vec3.h — minimal 3D vector and ray types for flaresim
// ============================================================================
#pragma once

#include <cmath>
#include <algorithm>

struct Vec3f
{
    float x, y, z;

    Vec3f() : x(0), y(0), z(0) {}
    Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    explicit Vec3f(float s) : x(s), y(s), z(s) {}

    Vec3f operator+(const Vec3f &v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3f operator-(const Vec3f &v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3f operator*(const Vec3f &v) const { return {x * v.x, y * v.y, z * v.z}; }
    Vec3f operator/(float s) const
    {
        float inv = 1.0f / s;
        return {x * inv, y * inv, z * inv};
    }
    Vec3f operator-() const { return {-x, -y, -z}; }

    Vec3f &operator+=(const Vec3f &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vec3f &operator-=(const Vec3f &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    Vec3f &operator*=(float s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    float operator[](int i) const { return (&x)[i]; }
    float &operator[](int i) { return (&x)[i]; }

    float length_sq() const { return x * x + y * y + z * z; }
    float length() const { return std::sqrt(length_sq()); }

    Vec3f normalized() const
    {
        float l = length();
        return (l > 1e-12f) ? (*this) * (1.0f / l) : Vec3f(0);
    }
};

inline float dot(const Vec3f &a, const Vec3f &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3f cross(const Vec3f &a, const Vec3f &b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline Vec3f operator*(float s, const Vec3f &v) { return v * s; }

// ---- Ray ----------------------------------------------------------------

struct Ray
{
    Vec3f origin;
    Vec3f dir; // must be normalized
};

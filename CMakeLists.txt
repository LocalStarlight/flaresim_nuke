cmake_minimum_required(VERSION 3.18)
project(NukeFlareSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# All compilation units (including CUDA) must emit position-independent code,
# which is required for shared libraries on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Target CUDA Toolkit 12.x (12.6 or 12.8).
# The plugin uses only basic CUDA features (atomicAdd, standard math) and
# requires nothing from CUDA 13.  Building against 12.x lowers the minimum
# driver requirement to 525+, covering the vast majority of users.
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90 120)  # Maxwell–Blackwell; requires CUDA 12.8+

# Static CUDA runtime — bakes cudart into the plugin binary so users do not
# need a CUDA runtime installed.  On Windows we keep Shared to avoid any
# potential CRT conflict with Nuke's /MD runtime.
if(WIN32)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)   # use cudart.dll; avoids LIBCMT conflict with Nuke's /MD
else()
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)   # bake cudart in; no runtime dependency on Linux
endif()

# Match Nuke's MSVC runtime — must be MultiThreaded DLL (/MD).
# Mixing runtimes causes heap corruption and hard-to-diagnose crashes.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# ---------------------------------------------------------------------------
# Paths — override on the cmake command line if Nuke is installed elsewhere
# ---------------------------------------------------------------------------
if(WIN32)
    set(NDK_ROOT "C:/Program Files/Nuke16.0v1/include"
        CACHE PATH "Nuke NDK include directory")
    set(NUKE_LIB_DIR "C:/Program Files/Nuke16.0v1"
        CACHE PATH "Nuke library directory (contains DDImage.lib / Ndk.lib)")
else()
    set(NDK_ROOT "/usr/local/Nuke16.0v1/include"
        CACHE PATH "Nuke NDK include directory")
    set(NUKE_LIB_DIR "/usr/local/Nuke16.0v1"
        CACHE PATH "Nuke library directory (contains libDDImage.so)")
endif()

# ---------------------------------------------------------------------------
# Plugin target
# ---------------------------------------------------------------------------
add_library(NukeFlareSim SHARED
    src/FlareSim.cpp
    src/ghost_cuda.cu
    src/lens.cpp
    src/trace.cpp
    src/ghost.cpp
    src/starburst.cpp
)

target_include_directories(NukeFlareSim PRIVATE
    src
    "${NDK_ROOT}"
)

if(WIN32)
    target_compile_definitions(NukeFlareSim PRIVATE
        NOMINMAX            # required by Nuke NDK (fnVC.h enforces this)
        _USE_MATH_DEFINES   # required by Nuke NDK (fnVC.h enforces this); exposes M_PI etc.
        WIN32_LEAN_AND_MEAN
    )
endif()

# CXX warning flags — platform-conditional.
# MSVC: treat warnings as informational only; keep /W3.
#   /wd4996 suppresses STL3DAllocator deprecation warnings that originate
#   inside Nuke's own DDImage headers (Attribute.h, Primitive.h, Allocators.h).
# GCC/Clang: standard warnings.
if(MSVC)
    target_compile_options(NukeFlareSim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /wd4996>
    )
else()
    target_compile_options(NukeFlareSim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wno-deprecated-declarations -fPIC>
    )
endif()

# CUDA: allow constexpr functions to be used on device without __device__ tags;
# required for some C++17 standard-library helpers used in device-side headers.
target_compile_options(NukeFlareSim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(WIN32)
    target_link_libraries(NukeFlareSim PRIVATE
        "${NUKE_LIB_DIR}/DDImage.lib"
        "${NUKE_LIB_DIR}/Ndk.lib"
    )
else()
    target_link_libraries(NukeFlareSim PRIVATE
        "${NUKE_LIB_DIR}/libDDImage.so"
    )
    # Not all Nuke versions ship a separate libNdk.so on Linux.
    if(EXISTS "${NUKE_LIB_DIR}/libNdk.so")
        target_link_libraries(NukeFlareSim PRIVATE
            "${NUKE_LIB_DIR}/libNdk.so"
        )
    endif()
endif()

# Nuke expects the plugin as a plain shared library with no "lib" prefix.
# CMAKE_SHARED_LIBRARY_SUFFIX gives ".dll" on Windows and ".so" on Linux.
set_target_properties(NukeFlareSim PROPERTIES
    OUTPUT_NAME "FlareSim"     # plugin name must match the registered node class name
    PREFIX ""
    SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
    CUDA_SEPARABLE_COMPILATION OFF
)

# ---------------------------------------------------------------------------
# Install helper — copies the plugin into the directory pointed to by
# the NUKE_PLUGIN_DIR cache variable if provided.
# Usage: cmake -DNUKE_PLUGIN_DIR="C:/Users/<username>/.nuke/plugins" ..
# ---------------------------------------------------------------------------
if(DEFINED NUKE_PLUGIN_DIR)
    install(TARGETS NukeFlareSim DESTINATION "${NUKE_PLUGIN_DIR}")
endif()

cmake_minimum_required(VERSION 3.18)
project(NukeFlareSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)  # Turing, Ampere, Ada, Hopper

# Match Nuke's MSVC runtime — must be MultiThreaded DLL (/MD).
# Mixing runtimes causes heap corruption and hard-to-diagnose crashes.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)   # use cudart.dll; avoids LIBCMT conflict with Nuke's /MD

# ---------------------------------------------------------------------------
# Paths — override on the cmake command line if Nuke is installed elsewhere
# ---------------------------------------------------------------------------
set(NDK_ROOT "C:/Program Files/Nuke16.0v1/include"
    CACHE PATH "Nuke NDK include directory")
set(NUKE_LIB_DIR "C:/Program Files/Nuke16.0v1"
    CACHE PATH "Nuke library directory (contains DDImage.lib / Ndk.lib)")

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

target_compile_definitions(NukeFlareSim PRIVATE
    NOMINMAX            # required by Nuke NDK (fnVC.h enforces this)
    _USE_MATH_DEFINES   # required by Nuke NDK (fnVC.h enforces this); exposes M_PI etc.
    WIN32_LEAN_AND_MEAN
)

# CXX: treat warnings as informational only; keep /W3
target_compile_options(NukeFlareSim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3>
)

# CUDA: allow constexpr functions to be used on device without __device__ tags;
# required for some C++17 standard-library helpers used in device-side headers.
target_compile_options(NukeFlareSim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(NukeFlareSim PRIVATE
    "${NUKE_LIB_DIR}/DDImage.lib"
    "${NUKE_LIB_DIR}/Ndk.lib"
)

# Nuke expects the plugin as a plain .dll with no "lib" prefix.
set_target_properties(NukeFlareSim PROPERTIES
    OUTPUT_NAME "FlareSim"     # DLL name must match the registered node class name
    PREFIX ""
    SUFFIX ".dll"
    CUDA_SEPARABLE_COMPILATION OFF
)

# ---------------------------------------------------------------------------
# Install helper — copies the DLL into the directory pointed to by
# the NUKE_PLUGIN_DIR cache variable if provided.
# Usage: cmake -DNUKE_PLUGIN_DIR="C:/Users/Steve/.nuke/plugins" ..
# ---------------------------------------------------------------------------
if(DEFINED NUKE_PLUGIN_DIR)
    install(TARGETS NukeFlareSim DESTINATION "${NUKE_PLUGIN_DIR}")
endif()

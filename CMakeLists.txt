cmake_minimum_required(VERSION 3.18)
project(NukeFlareSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# GPU architectures: Turing, Ampere, Ada, Hopper, Blackwell
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 120)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()
# Static on Linux: avoids conflicts with Nuke's bundled libcudart.so.11.
# Shared on Windows: the original default; avoids LIBCMT conflict with Nuke's /MD.
if(WIN32)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
else()
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
endif()

# ---------------------------------------------------------------------------
# Paths — override on the cmake command line if Nuke is installed elsewhere
# ---------------------------------------------------------------------------
if(WIN32)
    set(NDK_ROOT "C:/Program Files/Nuke16.0v1/include"
        CACHE PATH "Nuke NDK include directory")
    set(NUKE_LIB_DIR "C:/Program Files/Nuke16.0v1"
        CACHE PATH "Nuke library directory")
else()
    set(NDK_ROOT "$ENV{HOME}/Nuke/Nuke16.0v7/include"
        CACHE PATH "Nuke NDK include directory")
    set(NUKE_LIB_DIR "$ENV{HOME}/Nuke/Nuke16.0v7"
        CACHE PATH "Nuke library directory")
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

target_compile_definitions(NukeFlareSim PRIVATE
    NOMINMAX
    _USE_MATH_DEFINES
)
if(WIN32)
    target_compile_definitions(NukeFlareSim PRIVATE WIN32_LEAN_AND_MEAN)
endif()

# Platform-specific compile options
if(MSVC)
    target_compile_options(NukeFlareSim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /wd4996>
    )
else()
    target_compile_options(NukeFlareSim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wno-deprecated-declarations>
    )
endif()

# CUDA: allow constexpr functions to be used on device without __device__ tags
target_compile_options(NukeFlareSim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Link against Nuke libraries
if(WIN32)
    target_link_libraries(NukeFlareSim PRIVATE
        "${NUKE_LIB_DIR}/DDImage.lib"
        "${NUKE_LIB_DIR}/Ndk.lib"
    )
else()
    target_link_libraries(NukeFlareSim PRIVATE
        "${NUKE_LIB_DIR}/libDDImage.so"
        "${NUKE_LIB_DIR}/libNdk.so"
    )
endif()

# Nuke expects the plugin with no "lib" prefix.
# Windows: .dll, Linux: .so.
if(WIN32)
    set(FLARESIM_PLUGIN_EXT ".dll")
else()
    set(FLARESIM_PLUGIN_EXT ".so")
endif()
set_target_properties(NukeFlareSim PROPERTIES
    OUTPUT_NAME "FlareSim"
    PREFIX ""
    SUFFIX "${FLARESIM_PLUGIN_EXT}"
    CUDA_SEPARABLE_COMPILATION OFF
)

# ---------------------------------------------------------------------------
# Install helper
# ---------------------------------------------------------------------------
if(DEFINED NUKE_PLUGIN_DIR)
    install(TARGETS NukeFlareSim DESTINATION "${NUKE_PLUGIN_DIR}")
endif()

# Empyrean: the additive ambient synth with 20,000 partials per voice, on the GPU (docs/PLAN.md). The code keeps
# the folder's working name (namespace additiv, ADDITIV_ options and variables), as Noctuary's keeps `ambient`.
#
# Layout:
#   Core/          the engine (static library AdditivCore): the GPU cloud (CUDA), the instrument around it, and the
#                  modules copied from Noctuary (tuning, conductor, effects); no JUCE
#   Tools/probe    probe: real-time measurements, the listening comparisons, the motion scenes
#   Tools/render   empy_render: the instrument offline, for a night without a window
#   Tests/         additiv_selftest
#   Plugin/        JUCE VST3 + Standalone
#
# Built with CUDA 13.4 for the RTX 5090 (Blackwell, sm_120); the presets select CUDA 13.4 by its folder, whatever
# CUDA_PATH or the nvcc on the PATH say. Windows only for now; no macOS path (user, 11.10.2026).
cmake_minimum_required(VERSION 3.30)
project(Empyrean VERSION 0.13.1 LANGUAGES C CXX CUDA)

# The family's layout (01.10.2026): build/<preset>, bin/<preset>, dist/, work/ -- cmake/Family.cmake (the same file in
# every repository of the family), CMakePresets.json and build.ps1.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Family.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
add_compile_definitions(ADDITIV_VERSION="${PROJECT_VERSION}")

option(ADDITIV_BUILD_TOOLS "Build probe, empy_render and the tests" ON)
option(ADDITIV_BUILD_PLUGIN "Build the JUCE plugin (VST3 + Standalone)" ON)
option(ADDITIV_AVX2 "Build the C++ of the core with AVX2 + FMA (the effects' vector paths)" ON)
option(ADDITIV_STATIC_RUNTIME "Link the MSVC runtime statically (the release: no redistributable, no DLL beside the program)" OFF)
if(ADDITIV_STATIC_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
set(ADDITIV_JUCE_TAG "9.0.1" CACHE STRING "JUCE git tag to fetch when no local checkout is found")
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
endif()

find_package(CUDAToolkit 13.4 REQUIRED)

# The engine: the GPU cloud (kernels, shapes, landscapes, motions), the instrument (Engine), and Noctuary's tuning,
# conductor and effects. Pictures through the Windows Imaging Component; FLAC through dr_flac, vendored from
# Noctuary's ThirdParty/dr_libs (public domain or MIT-0).
add_library(AdditivCore STATIC
    Core/src/Cloud.cu
    Core/src/Shapes.cpp
    Core/src/Landscape.cpp
    Core/src/Motion.cpp
    Core/src/Params.cpp
    Core/src/Tuning.cpp
    Core/src/Effects.cpp
    Core/src/Modulation.cpp
    Core/src/Presets.cpp
    Core/src/Morph.cpp
    Core/src/Journeys.cpp
    Core/src/PresetMap.cpp
    Core/src/Route.cpp
    Core/src/Engine.cpp)
target_include_directories(AdditivCore PUBLIC Core/include PRIVATE ThirdParty/dr_libs)
target_compile_definitions(AdditivCore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:_WIN32_WINNT=0x0A00 UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS>)
target_compile_options(AdditivCore PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --Werror cross-execution-space-call -Xcompiler=/wd4211>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${ADDITIV_AVX2}>>:/arch:AVX2>
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
target_link_libraries(AdditivCore PUBLIC CUDA::cudart_static windowscodecs ole32 avrt)
set_target_properties(AdditivCore PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ADDITIV_BUILD_TOOLS)
    enable_testing()
    family_doccheck()   # every entity documented (cmake/Family.cmake, 01.10.2026)

    # probe: real-time measurements, the listening comparisons and the motion scenes.
    add_executable(probe Tools/probe/main.cpp)
    target_compile_definitions(probe PRIVATE _WIN32_WINNT=0x0A00 UNICODE _UNICODE)
    target_compile_options(probe PRIVATE /W4 /permissive-)
    target_link_libraries(probe PRIVATE AdditivCore avrt winmm)
    family_bin(probe)

    # empy_render: the instrument offline -- the conductor alone for minutes or hours, or a chord.
    add_executable(empy_render Tools/render/main.cpp)
    target_compile_options(empy_render PRIVATE /W4 /permissive-)
    target_link_libraries(empy_render PRIVATE AdditivCore)
    # An Engine lives on main's stack in the tools and the self-test; with its modulators (Lenia's field
    # among them) forty of them in one frame passed Windows' 1 MB: 8 MB, as the family's Linux builds have.
    target_link_options(empy_render PRIVATE /STACK:8388608)
    family_bin(empy_render)

    # additiv_presets: every factory preset's level, for make_presets.py to bring the bank to one level.
    add_executable(additiv_presets Tools/presets/main.cpp)
    target_compile_options(additiv_presets PRIVATE /W4 /permissive-)
    target_link_libraries(additiv_presets PRIVATE AdditivCore)
    target_link_options(additiv_presets PRIVATE /STACK:8388608)
    family_bin(additiv_presets)

    add_executable(additiv_selftest Tests/selftest.cpp)
    target_compile_options(additiv_selftest PRIVATE /W4 /permissive-)
    target_link_libraries(additiv_selftest PRIVATE AdditivCore)
    target_link_options(additiv_selftest PRIVATE /STACK:8388608)
    add_test(NAME selftest COMMAND additiv_selftest)
    add_test(NAME probe COMMAND probe selftest)
endif()

# The plugin. JUCE from ThirdParty/JUCE, else from the sibling Phosphene's checkout, else fetched (cmake/Family.cmake).
# Nothing under Core/ includes JUCE.
if(ADDITIV_BUILD_PLUGIN)
    family_juce(${ADDITIV_JUCE_TAG})
    add_subdirectory(Plugin)
    # Tracktion's pluginval (unpacked into ThirdParty/pluginval or the sibling Ephemeris's, not in the repository) at its
    # highest strictness, where it is there -- as in Totality.
    if(ADDITIV_BUILD_TOOLS)
        set(ADDITIV_PLUGINVAL "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/pluginval/pluginval.exe")
        if(NOT EXISTS "${ADDITIV_PLUGINVAL}")
            set(ADDITIV_PLUGINVAL "${CMAKE_CURRENT_SOURCE_DIR}/../BerlinSchoolGenerator/ThirdParty/pluginval/pluginval.exe")
        endif()
        if(EXISTS "${ADDITIV_PLUGINVAL}")
            add_test(NAME pluginval COMMAND "${ADDITIV_PLUGINVAL}" --strictness-level 10 --timeout-ms 900000
                     --validate "$<TARGET_FILE_DIR:Empyrean_VST3>/../..")
            set_tests_properties(pluginval PROPERTIES LABELS "audio;slow")
        endif()
    endif()
endif()

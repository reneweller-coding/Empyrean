# AdditivAmbientSynth: the additive ambient synth with tens of thousands of partials per voice, on the GPU.
#
# Built with CUDA 13.4 for the RTX 5090 (Blackwell, sm_120); the presets select CUDA 13.4's Visual Studio
# integration, whatever CUDA_PATH or the nvcc on the PATH say. No macOS path.
cmake_minimum_required(VERSION 3.30)
project(AdditivAmbientSynth VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(CUDAToolkit 13.4 REQUIRED)

# The programs land in bin/<preset> (the family's layout); a Debug build beside it in bin/<preset>-Debug.
set(FAMILY_BIN_DIR "" CACHE PATH "where the programs are put (bin/<preset>); empty = the build tree")

# The GPU engine: the partial cloud's kernels and the graph that runs them, the factory envelope
# shapes, and the landscapes (pictures through the Windows Imaging Component, FLAC through dr_flac,
# vendored from Noctuary's ThirdParty/dr_libs: public domain or MIT-0).
add_library(additiv_gpu STATIC Gpu/src/Cloud.cu Gpu/src/Shapes.cpp Gpu/src/Landscape.cpp)
target_include_directories(additiv_gpu PUBLIC Gpu/include PRIVATE ThirdParty/dr_libs)
target_compile_definitions(additiv_gpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:_WIN32_WINNT=0x0A00 UNICODE _UNICODE>)
target_compile_options(additiv_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --Werror cross-execution-space-call>)
target_link_libraries(additiv_gpu PUBLIC CUDA::cudart_static windowscodecs ole32)

# probe: real-time measurements and the listening comparison (Tools/probe/main.cpp).
add_executable(probe Tools/probe/main.cpp)
target_compile_definitions(probe PRIVATE _WIN32_WINNT=0x0A00 UNICODE _UNICODE)
target_compile_options(probe PRIVATE /W4 /permissive-)
target_link_libraries(probe PRIVATE additiv_gpu avrt winmm)
if(FAMILY_BIN_DIR)
    set_target_properties(probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${FAMILY_BIN_DIR}$<$<CONFIG:Debug>:-Debug>")
endif()

enable_testing()
add_test(NAME selftest COMMAND probe selftest)

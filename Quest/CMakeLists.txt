# Empyrean for Meta Quest -- a native OpenXR app (NativeActivity, GLES 3), no game engine, no synthesizer: the head and
# the hands to the desktop's Empyrean (Phase 30). Configured by build_apk.ps1 with the NDK toolchain; see Quest/README.md.
# After Totality's Quest/CMakeLists.txt, without its core and its audio.
cmake_minimum_required(VERSION 3.22)
project(EmpyreanQuest LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EMPY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(EMPY_THIRDPARTY "${EMPY_ROOT}/ThirdParty" CACHE PATH "the OpenXR loader (Quest/fetch_thirdparty.ps1)")

# Khronos OpenXR loader for Android: the prefab layout of the Maven AAR (headers + libopenxr_loader.so).
set(OXR "${EMPY_THIRDPARTY}/openxr-loader/prefab/modules")
if(NOT EXISTS "${OXR}/openxr_loader/libs/android.${ANDROID_ABI}/libopenxr_loader.so")
    message(FATAL_ERROR "OpenXR loader missing under ${EMPY_THIRDPARTY}: run Quest/fetch_thirdparty.ps1")
endif()
add_library(openxr_loader SHARED IMPORTED)
set_target_properties(openxr_loader PROPERTIES
    IMPORTED_LOCATION "${OXR}/openxr_loader/libs/android.${ANDROID_ABI}/libopenxr_loader.so"
    INTERFACE_INCLUDE_DIRECTORIES "${OXR}/headers/include")

# The NDK's native activity glue.
add_library(native_app_glue STATIC "${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c")
target_include_directories(native_app_glue PUBLIC "${ANDROID_NDK}/sources/android/native_app_glue")

add_library(empyquest SHARED src/main.cpp)
target_compile_definitions(empyquest PRIVATE XR_USE_PLATFORM_ANDROID XR_USE_GRAPHICS_API_OPENGL_ES)
target_link_libraries(empyquest PRIVATE openxr_loader native_app_glue android log EGL GLESv3)
# ANativeActivity_onCreate lives in the glue library and nothing in our code references it.
target_link_options(empyquest PRIVATE -u ANativeActivity_onCreate)
target_compile_options(empyquest PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)

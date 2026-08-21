cmake_minimum_required(VERSION 3.16)
project(MegaVitFrontend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(TFDL_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(TFDL_SDK_ROOT "${TFDL_SDK_ROOT}" CACHE PATH "TFDL2 SDK root")
option(MEGAVIT_WITH_FFMPEG "Build the libavcodec/libavformat decoder" ON)
option(MEGAVIT_WITH_TFDL "Build the mage_vit.fb runtime adapter" ON)
option(MEGAVIT_BUILD_TESTS "Build dependency-free frontend tests" ON)

add_library(megavit_core STATIC
    src/codec_frontend.cpp
    src/prompt_assembler.cpp
    src/rope.cpp)
target_include_directories(megavit_core PUBLIC include)
target_compile_options(megavit_core PRIVATE -Wall -Wextra -Wpedantic)

set(MEGAVIT_FFMPEG_SOURCE src/ffmpeg_decoder_stub.cpp)
if(MEGAVIT_WITH_FFMPEG)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFMPEG QUIET IMPORTED_TARGET
            libavformat libavcodec libavutil libswscale)
    endif()
    if(FFMPEG_FOUND)
        set(MEGAVIT_FFMPEG_SOURCE src/ffmpeg_decoder.cpp)
        message(STATUS "Mage-Vit: FFmpeg decoder enabled")
    else()
        message(WARNING
            "Mage-Vit: FFmpeg development packages not found; building the "
            "synthetic/stub decoder. Install libavformat/libavcodec/libavutil/"
            "libswscale development packages to enable video input.")
    endif()
endif()
add_library(megavit_decoder STATIC ${MEGAVIT_FFMPEG_SOURCE})
target_include_directories(megavit_decoder PUBLIC include)
target_link_libraries(megavit_decoder PUBLIC megavit_core)
if(FFMPEG_FOUND)
    target_link_libraries(megavit_decoder PRIVATE PkgConfig::FFMPEG)
endif()

set(MEGAVIT_VISION_SOURCE src/vision_runner_stub.cpp)
if(MEGAVIT_WITH_TFDL AND
   EXISTS "${TFDL_SDK_ROOT}/include/TFDL2_C_API.h" AND
   EXISTS "${TFDL_SDK_ROOT}/lib/libTFDL2_LITE_C_API.so")
    set(MEGAVIT_VISION_SOURCE src/vision_runner.cpp)
    set(MEGAVIT_TFDL_ENABLED ON)
    message(STATUS "Mage-Vit: TFDL runtime adapter enabled")
else()
    if(MEGAVIT_WITH_TFDL)
        message(WARNING "Mage-Vit: TFDL runtime library not found; building a stub adapter")
    else()
        message(STATUS "Mage-Vit: TFDL runtime adapter disabled")
    endif()
endif()
add_library(megavit_vision STATIC ${MEGAVIT_VISION_SOURCE})
target_include_directories(megavit_vision PUBLIC include)
target_link_libraries(megavit_vision PUBLIC megavit_core)
if(MEGAVIT_TFDL_ENABLED)
    target_include_directories(megavit_vision PRIVATE "${TFDL_SDK_ROOT}/include")
    target_link_libraries(megavit_vision PUBLIC
        "${TFDL_SDK_ROOT}/lib/libTFDL2_LITE_C_API.so")
endif()

add_executable(megavit_frontend src/main.cpp)
target_link_libraries(megavit_frontend PRIVATE
    megavit_core megavit_decoder megavit_vision)
if(FFMPEG_FOUND)
    target_compile_definitions(megavit_frontend PRIVATE MEGAVIT_HAS_FFMPEG=1)
else()
    target_compile_definitions(megavit_frontend PRIVATE MEGAVIT_HAS_FFMPEG=0)
endif()
if(MEGAVIT_TFDL_ENABLED)
    target_compile_definitions(megavit_frontend PRIVATE MEGAVIT_HAS_TFDL=1)
    set_target_properties(megavit_frontend PROPERTIES
        BUILD_RPATH "${TFDL_SDK_ROOT}/lib")
else()
    target_compile_definitions(megavit_frontend PRIVATE MEGAVIT_HAS_TFDL=0)
endif()

if(MEGAVIT_BUILD_TESTS)
    enable_testing()
    add_executable(megavit_frontend_test tests/test_frontend.cpp)
    target_link_libraries(megavit_frontend_test PRIVATE megavit_core)
    add_test(NAME megavit_frontend_test COMMAND megavit_frontend_test)
endif()

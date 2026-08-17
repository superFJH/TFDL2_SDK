cmake_minimum_required(VERSION 3.10)
project(TFDLAddOn LANGUAGES CXX)

aux_source_directory(./ addon)

add_library(TFDLAddOn SHARED ${addon} json11.cpp)
target_include_directories(TFDLAddOn PRIVATE ${CMAKE_SOURCE_DIR}/../include)

# These CPU fallback kernels are latency-critical. Keep
# their release optimization independent of the caller's global CMake build
# type; the SDK's historical addon build directory is often configured with
# an empty CMAKE_BUILD_TYPE.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/ArmCausalMaskSoftmax.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ApplyRope.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RMSNorm.cpp"
        PROPERTIES COMPILE_OPTIONS "-O3;-fno-math-errno"
    )
endif()

if(APPLE)
    # Developer-only build on macOS. The SDK ships Linux .so files, so do not
    # try to link them here; leave TFDL2 symbols to be resolved by the loader.
    target_link_options(TFDLAddOn PRIVATE "-undefined" "dynamic_lookup")
else()
    target_link_directories(TFDLAddOn PRIVATE ${CMAKE_SOURCE_DIR}/../lib)
    target_link_libraries(TFDLAddOn PRIVATE TFDL2_LITE_C_API)
endif()

# 启用 aarch64 FP16 NEON 向量指令 (ARMv8.2-A)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(TFDLAddOn PRIVATE -march=armv8.2-a+fp16)
endif()

# 启用 OpenMP 多线程
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(TFDLAddOn PUBLIC OpenMP::OpenMP_CXX)
endif()

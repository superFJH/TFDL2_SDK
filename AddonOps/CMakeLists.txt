aux_source_directory(./ addon)

add_library(TFDLAddOn SHARED ${addon} json11.cpp)
target_include_directories(TFDLAddOn PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_directories(TFDLAddOn PRIVATE ${CMAKE_SOURCE_DIR}/../lib)
target_link_libraries(TFDLAddOn TFDL2_LITE_C_API)

# 启用 aarch64 FP16 NEON 向量指令 (ARMv8.2-A)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(TFDLAddOn PRIVATE -march=armv8.2-a+fp16)
endif()

# 启用 OpenMP 多线程
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(TFDLAddOn PUBLIC OpenMP::OpenMP_CXX)
endif()


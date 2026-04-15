 cmake_minimum_required(VERSION 3.2)
project(Benchmark)

set(CMAKE_CXX_STANDARD 14)

add_executable(Benchmark Benchmark.cpp json11.cpp)
target_include_directories(Benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_directories(Benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
target_link_directories(Benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/CV_NPU10T)
target_link_libraries(Benchmark TFDL2_LITE_C_API TFCV tfg tfgs tfdec mk_api)

cmake_minimum_required(VERSION 3.2)
project(Classification)

set(CMAKE_CXX_STANDARD 14)

add_executable(Classification main.cpp)
target_include_directories(Classification PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_directories(Classification PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
target_link_directories(Classification PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/CV_NPU10T)
target_link_libraries(Classification TFDL2_LITE_C_API TFCV tfg tfgs tfdec mk_api)

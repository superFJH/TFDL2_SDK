cmake_minimum_required(VERSION 3.2)
project(RtspServer)

# compiler setting
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)


# include
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/CV_NPU10T)
add_executable(RtspServer main.cpp)
target_include_directories(RtspServer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(RtspServer TFDL2_LITE_C_API TFCV tfg tfgs tfdec mk_api)
target_link_libraries(RtspServer ${OpenCV_LIBS})

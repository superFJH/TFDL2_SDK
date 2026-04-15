aux_source_directory(./ addon)

add_library(TFDLAddOn SHARED ${addon} json11.cpp)
target_include_directories(TFDLAddOn PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_directories(TFDLAddOn PRIVATE ${CMAKE_SOURCE_DIR}/../lib)
target_link_libraries(TFDLAddOn TFDL2_LITE_C_API)


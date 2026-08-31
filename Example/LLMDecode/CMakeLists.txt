cmake_minimum_required(VERSION 3.16)

project(LLMDecode LANGUAGES C CXX)

option(LLMDECODE_KLEIDIAI "Enable llama.cpp's ARM KleidiAI CPU kernels" ON)
option(LLMDECODE_NATIVE "Enable CPU-specific compiler tuning" ON)
option(LLMDECODE_LTO "Enable link-time optimization for the decoder build" ON)

if(LLMDECODE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LLMDECODE_IPO_SUPPORTED OUTPUT LLMDECODE_IPO_ERROR LANGUAGES C CXX)
    if(LLMDECODE_IPO_SUPPORTED)
        # Set this before adding llama.cpp so its CPU kernels and their callers
        # are linked as one optimized unit.  The build script keeps C/C++ GCC
        # versions matched when it enables this option.
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(WARNING "LLMDECODE_LTO requested but unavailable: ${LLMDECODE_IPO_ERROR}")
    endif()
endif()

# We intentionally build only the library.  The project executable below owns
# the heterogeneous-prefill ABI, so pulling in llama-server would not make its
# incompatible prompt-prefill path accidentally available.
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI ${LLMDECODE_KLEIDIAI} CACHE BOOL "" FORCE)
set(GGML_NATIVE ${LLMDECODE_NATIVE} CACHE BOOL "" FORCE)
# Every llama.cpp shared object is shipped beside the executable in the Python
# wheel.  Apply the same relocatable rpath to the vendored subproject before
# it creates its targets; otherwise CMake records this machine's build path
# in indirect ggml dependencies and the worker exits with 127 on deployment.
set(CMAKE_BUILD_RPATH "\$ORIGIN")
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "\$ORIGIN")
# The vendored directory intentionally has no nested .git metadata.  Pin the
# build banner independently so it cannot inherit this SDK repository's SHA.
set(LLAMA_BUILD_COMMIT "9723942" CACHE STRING "" FORCE)

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

add_executable(llmdecode src/llmdecode.cpp)
target_compile_features(llmdecode PRIVATE cxx_std_17)
target_link_libraries(llmdecode PRIVATE llama)
set_target_properties(llmdecode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    # The Python wheel carries this executable and the matching llama/ggml
    # shared objects together.  Never bake a developer build directory into
    # the portable runtime loader path.
    BUILD_RPATH "\$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN")

# A deliberately small wrapper around llama.cpp's public quantization API.
# Keeping it here lets a deployment build produce Q4_K_M without bringing in
# llama.cpp's prompt/server applications, which do not accept our external KV
# prefix ABI.
add_executable(llmquantize src/llmquantize.cpp)
target_compile_features(llmquantize PRIVATE cxx_std_17)
target_link_libraries(llmquantize PRIVATE llama)
set_target_properties(llmquantize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    BUILD_RPATH "\$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN")

# Isolated proof-of-concept for external-KV continuous batching.  It is not
# shipped in the deployment wheel: the program creates deterministic distinct
# KV prefixes, imports them into different llama sequence IDs, then executes
# an M=B decode batch after the remaining requests arrive.
add_executable(llmdecode_batch_probe src/llmdecode_batch_probe.cpp)
target_compile_features(llmdecode_batch_probe PRIVATE cxx_std_17)
target_link_libraries(llmdecode_batch_probe PRIVATE llama)
set_target_properties(llmdecode_batch_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    BUILD_RPATH "\$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN")

install(TARGETS llmdecode llmquantize RUNTIME DESTINATION bin)

// Minimal weight-only GGUF quantizer for the LLMDecode deployment project.
//
// Q4_K_M does not require an activation calibration set: llama.cpp applies
// block-wise weight quantization directly to an F16/BF16 GGUF source.

#include <cstdlib>
#include <iostream>
#include <string>

#include "llama.h"

namespace {

void usage(const char * program) {
    std::cerr
        << "Usage: " << program << " INPUT_F16_OR_BF16.gguf OUTPUT_Q4_K_M.gguf [--threads N]\\n"
        << "\\n"
        << "Quantizes weights directly to llama.cpp's Q4_K_M format.  Do not use a\\n"
        << "Q8 GGUF input: quantizing from the original F16/BF16 source avoids an\\n"
        << "extra source-quantization error.\\n";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }

    const std::string input = argv[1];
    const std::string output = argv[2];
    int threads = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
            if (threads <= 0) {
                std::cerr << "--threads must be positive\\n";
                return 2;
            }
        } else if (option == "--help" || option == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << option << '\n';
            usage(argv[0]);
            return 2;
        }
    }

    llama_backend_init();
    auto params = llama_model_quantize_default_params();
    params.ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
    params.nthread = threads;
    const uint32_t status = llama_model_quantize(input.c_str(), output.c_str(), &params);
    llama_backend_free();
    return static_cast<int>(status);
}

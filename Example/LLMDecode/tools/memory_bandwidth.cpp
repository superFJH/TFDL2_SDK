// Sequential DRAM-read benchmark for decoder weight-streaming analysis.
//
// The buffer is intentionally much larger than the shared cache.  It is not
// a synthetic peak benchmark: every OpenMP worker reads a contiguous shard,
// which approximates the access pattern of a CPU decoder scanning GGUF
// tensors once per generated token.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <omp.h>

namespace {

size_t parse_size(const std::string & value) {
    const auto multiplier_at = value.find_first_not_of("0123456789");
    const auto number = std::stoull(value.substr(0, multiplier_at));
    const std::string suffix = multiplier_at == std::string::npos ? "B" : value.substr(multiplier_at);
    if (suffix == "B") return number;
    if (suffix == "K" || suffix == "KiB") return number * 1024ULL;
    if (suffix == "M" || suffix == "MiB") return number * 1024ULL * 1024ULL;
    if (suffix == "G" || suffix == "GiB") return number * 1024ULL * 1024ULL * 1024ULL;
    throw std::invalid_argument("unsupported size suffix: " + suffix);
}

void usage(const char * program) {
    std::cerr << "Usage: " << program << " [--bytes 4GiB] [--passes 4] [--threads N]\n";
}

} // namespace

int main(int argc, char ** argv) {
    size_t bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    int passes = 4;
    int threads = omp_get_max_threads();

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--bytes" && i + 1 < argc) {
            bytes = parse_size(argv[++i]);
        } else if (option == "--passes" && i + 1 < argc) {
            passes = std::stoi(argv[++i]);
        } else if (option == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (option == "--help" || option == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (bytes < 64 || bytes % sizeof(uint64_t) != 0 || passes <= 0 || threads <= 0) {
        throw std::invalid_argument("bytes must be a positive multiple of 64, passes and threads must be positive");
    }

    void * raw = nullptr;
    if (posix_memalign(&raw, 64, bytes) != 0 || raw == nullptr) {
        throw std::bad_alloc();
    }
    auto * data = static_cast<uint64_t *>(raw);
    const size_t count = bytes / sizeof(*data);

    // Physical allocation and a deterministic payload. This happens before
    // timing so page faults and first-touch writes do not inflate read rate.
#pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < count; ++i) {
        data[i] = 0x9e3779b97f4a7c15ULL ^ static_cast<uint64_t>(i);
    }

    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int pass = 0; pass < passes; ++pass) {
        uint64_t pass_checksum = 0;
#pragma omp parallel for reduction(^ : pass_checksum) schedule(static) num_threads(threads)
        for (size_t i = 0; i < count; ++i) {
            pass_checksum ^= data[i];
        }
        checksum ^= pass_checksum;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double gib_read = static_cast<double>(bytes) * passes / (1024.0 * 1024.0 * 1024.0);
    const double gb_read = static_cast<double>(bytes) * passes / 1.0e9;

    std::cout << std::fixed << std::setprecision(3)
              << "bytes=" << bytes
              << " passes=" << passes
              << " threads=" << threads
              << " seconds=" << seconds
              << " GiB/s=" << gib_read / seconds
              << " GB/s=" << gb_read / seconds
              << " checksum=" << checksum << '\n';
    std::free(raw);
    return 0;
}

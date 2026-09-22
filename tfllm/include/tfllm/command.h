#pragma once
#include "tfllm/model.h"

namespace tfllm {
enum class RegionKind : uint8_t { Input, Output, Weight, Workspace, Kv };
enum class AddressPart : uint8_t { Low32, High8, Full40 };
struct Region { std::string name; RegionKind kind = RegionKind::Workspace; uint64_t bytes = 0; uint32_t alignment = 64; };
struct Image { uint32_t region = 0; std::vector<uint8_t> bytes; };
struct Relocation {
    uint32_t image = 0;
    uint64_t byte_offset = 0;
    uint32_t region = 0;
    uint64_t addend = 0, span = 1;
    AddressPart part = AddressPart::Low32;
    uint8_t right_shift = 0, bit_offset = 0, bit_count = 32;
    uint32_t alignment = 1;
};
struct Program {
    std::string name, command_abi;
    uint32_t core_kind = 16, query_bucket = 0, key_bucket = 0;
    std::vector<Region> regions;
    std::vector<Image> images;
    std::vector<Relocation> relocations;
    std::vector<uint8_t> heads;
    void Validate() const;
};
struct Binding { uint64_t physical = 0, bytes = 0; };
struct BoundProgram { uint32_t address_high = 0; std::vector<Image> images; };
// Always patches a private copy of the immutable template. Validates the full
// referenced spans, the 40-bit bus and the shared 4-GiB address window.
BoundProgram Bind(const Program& program, const std::vector<Binding>& bindings,
                  const std::string& expected_abi, uint32_t core_kind);
}

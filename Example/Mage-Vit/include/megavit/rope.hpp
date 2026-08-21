#pragma once

#include "megavit/types.hpp"

#include <vector>

namespace megavit {

struct RopeTables {
    int head_dim = 64;
    std::vector<float> sin;
    std::vector<float> cos;
};

// Exact Mage-VL vision RoPE: head_dim=64, half-frequency T:H:W split 4:6:6,
// concatenate the D/2 frequencies with themselves, then use interleaved
// adjacent-pair rotation in ApplyRope.
RopeTables BuildMageVisionRope(
    const std::vector<PatchPosition>& positions,
    float theta = 10000.0f,
    int head_dim = 64);

}  // namespace megavit

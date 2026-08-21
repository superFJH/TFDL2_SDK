#include "megavit/rope.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace megavit {
namespace {

std::vector<float> InverseFrequencies(int size, float theta) {
    std::vector<float> result(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i) {
        result[static_cast<size_t>(i)] =
            1.0f / std::pow(theta, static_cast<float>(i) / size);
    }
    return result;
}

}  // namespace

RopeTables BuildMageVisionRope(
    const std::vector<PatchPosition>& positions,
    float theta,
    int head_dim) {
    if (head_dim <= 0 || head_dim % 32 != 0) {
        throw std::invalid_argument(
            "Mage vision head_dim must be positive and divisible by 32");
    }
    const int half = head_dim / 2;
    const int unit = half / 16;
    const int t_size = 4 * unit;
    const int h_size = 6 * unit;
    const int w_size = 6 * unit;
    const auto inv_t = InverseFrequencies(t_size, theta);
    const auto inv_h = InverseFrequencies(h_size, theta);
    const auto inv_w = InverseFrequencies(w_size, theta);

    RopeTables tables;
    tables.head_dim = head_dim;
    tables.sin.resize(positions.size() * static_cast<size_t>(head_dim));
    tables.cos.resize(positions.size() * static_cast<size_t>(head_dim));
    for (size_t n = 0; n < positions.size(); ++n) {
        std::vector<float> half_freq;
        half_freq.reserve(static_cast<size_t>(half));
        for (float f : inv_t) half_freq.push_back(positions[n].t * f);
        for (float f : inv_h) half_freq.push_back(positions[n].h * f);
        for (float f : inv_w) half_freq.push_back(positions[n].w * f);
        for (int d = 0; d < head_dim; ++d) {
            const float angle = half_freq[static_cast<size_t>(d % half)];
            const size_t index = n * static_cast<size_t>(head_dim) + d;
            tables.sin[index] = std::sin(angle);
            tables.cos[index] = std::cos(angle);
        }
    }
    return tables;
}

}  // namespace megavit

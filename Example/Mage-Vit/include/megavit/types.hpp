#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace megavit {

struct PatchPosition {
    int32_t t = 0;
    int32_t h = 0;
    int32_t w = 0;
};

struct ScoreGrid {
    int width = 0;
    int height = 0;
    int cell_width = 0;
    int cell_height = 0;
    std::vector<float> values;

    bool valid() const {
        return width > 0 && height > 0 && cell_width > 0 && cell_height > 0 &&
               values.size() == static_cast<size_t>(width * height);
    }

    float at(int y, int x) const {
        if (x < 0 || y < 0 || x >= width || y >= height || !valid()) {
            return 0.0f;
        }
        return values[static_cast<size_t>(y * width + x)];
    }
};

struct DecodedFrame {
    int frame_index = 0;
    double timestamp_seconds = 0.0;
    int width = 0;
    int height = 0;
    bool key_frame = false;
    // Packed RGB24, row-major, HWC.
    std::vector<uint8_t> rgb;
    // Optional block-level codec score. A patched FFmpeg implementation should
    // put macroblock/CTU bitcost here. The upstream-FFmpeg fallback stores a
    // motion-vector score grid instead.
    ScoreGrid codec_score;

    void validate() const {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("decoded frame has invalid dimensions");
        }
        const size_t expected = static_cast<size_t>(width) * height * 3;
        if (rgb.size() != expected) {
            throw std::invalid_argument("decoded frame RGB byte count does not match dimensions");
        }
    }
};

struct Canvas {
    int width = 0;
    int height = 0;
    double timestamp_seconds = 0.0;
    // Packed RGB24, row-major, HWC.
    std::vector<uint8_t> rgb;
    // Patch positions are in Mage's 2x2 block order:
    // TL, TR, BL, BR for every selected block.
    std::vector<PatchPosition> patch_positions;
    // Parallel to patch_positions. Kept outside PatchPosition because only
    // integer (t,h,w) enters the vision RoPE graph.
    std::vector<double> patch_timestamps;
};

struct VisionEmbeddingBatch {
    int hidden_size = 0;
    std::vector<int> tokens_per_canvas;
    std::vector<float> values;

    size_t token_count() const {
        size_t count = 0;
        for (int n : tokens_per_canvas) {
            if (n < 0) {
                throw std::runtime_error("negative token count in vision result");
            }
            count += static_cast<size_t>(n);
        }
        return count;
    }
};

}  // namespace megavit

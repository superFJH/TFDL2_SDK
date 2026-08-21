#include "megavit/prompt_assembler.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace megavit {

PromptPlan BuildPromptPlan(
    const std::vector<Canvas>& canvases,
    int spatial_merge_size,
    int timestamp_decimals) {
    if (spatial_merge_size <= 0) {
        throw std::invalid_argument("spatial_merge_size must be positive");
    }
    const int merge_factor = spatial_merge_size * spatial_merge_size;
    PromptPlan plan;
    int offset = 0;
    bool have_previous = false;
    int previous_t = 0;
    double previous_timestamp = 0.0;
    for (const auto& canvas : canvases) {
        if (canvas.patch_positions.size() % static_cast<size_t>(merge_factor) != 0) {
            throw std::invalid_argument(
                "canvas patch count is not divisible by spatial merge factor");
        }
        if (!canvas.patch_timestamps.empty() &&
            canvas.patch_timestamps.size() != canvas.patch_positions.size()) {
            throw std::invalid_argument(
                "patch timestamps must be empty or parallel to patch positions");
        }
        for (size_t patch = 0; patch < canvas.patch_positions.size();
             patch += static_cast<size_t>(merge_factor)) {
            const int t = canvas.patch_positions[patch].t;
            const double timestamp = canvas.patch_timestamps.empty()
                ? canvas.timestamp_seconds
                : canvas.patch_timestamps[patch];
            for (int i = 1; i < merge_factor; ++i) {
                const size_t member = patch + static_cast<size_t>(i);
                if (canvas.patch_positions[member].t != t) {
                    throw std::invalid_argument(
                        "one spatial-merge block contains several source frames");
                }
                if (!canvas.patch_timestamps.empty() &&
                    std::abs(canvas.patch_timestamps[member] - timestamp) > 1e-9) {
                    throw std::invalid_argument(
                        "one spatial-merge block contains several timestamps");
                }
            }
            const bool same_run = have_previous && previous_t == t &&
                std::abs(previous_timestamp - timestamp) < 1e-9;
            if (same_run) {
                ++plan.spans.back().token_count;
            } else {
                plan.spans.push_back({timestamp, 1, offset});
            }
            have_previous = true;
            previous_t = t;
            previous_timestamp = timestamp;
            ++offset;
        }
    }
    std::ostringstream content;
    for (const auto& span : plan.spans) {
        content << '<' << std::fixed << std::setprecision(timestamp_decimals)
                << span.timestamp_seconds << " seconds>"
                << "<|vision_start|>";
        for (int i = 0; i < span.token_count; ++i) {
            content << "<|image_pad|>";
        }
        content << "<|vision_end|>\n";
    }
    plan.vision_content = content.str();
    plan.total_visual_tokens = offset;
    return plan;
}

}  // namespace megavit

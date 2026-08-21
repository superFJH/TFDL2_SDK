#pragma once

#include "megavit/types.hpp"

#include <string>
#include <vector>

namespace megavit {

struct VisionSpan {
    double timestamp_seconds = 0.0;
    int token_count = 0;
    int embedding_offset = 0;
};

struct PromptPlan {
    std::string vision_content;
    std::vector<VisionSpan> spans;
    int total_visual_tokens = 0;
};

// Produces only the multimodal content fragment. The Qwen tokenizer/chat
// template remains a Host/CPU/GPU responsibility and is intentionally not
// embedded in the TFDL graph.
PromptPlan BuildPromptPlan(
    const std::vector<Canvas>& canvases,
    int spatial_merge_size = 2,
    int timestamp_decimals = 1);

}  // namespace megavit

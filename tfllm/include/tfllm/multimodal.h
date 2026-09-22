#pragma once
#include "tfllm/activation.h"
#include <array>

namespace tfllm {
using Position3 = std::array<int32_t,3>; // T, H, W, independently of the KV slot.
struct VisualTokens {
    size_t height=0,width=0,hidden=0;
    Activation embedding;
    std::vector<Activation> deepstack;
    // Stable content identity including preprocessing and vision model.
    // Callers must not reuse this identity after modifying any feature.
    std::string identity;
    size_t Rows() const {return height*width;}
};
struct VisualSpan {size_t first=0; std::shared_ptr<const VisualTokens> image;bool bidirectional=false;};
struct PromptInput {
    bool interleaved=true; // Qwen3; Qwen2 uses contiguous MRoPE sections.
    std::array<int32_t,3> rope_sections{};
    std::vector<Position3> positions;
    std::vector<VisualSpan> images;
    Position3 Position(size_t token) const;
    const VisualSpan* Image(size_t token) const;
    bool Same(size_t token,const PromptInput* other) const;
    void Validate(size_t tokens,const Config&) const;
};
}

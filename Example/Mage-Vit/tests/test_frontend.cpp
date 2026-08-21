#include "megavit/codec_frontend.hpp"
#include "megavit/prompt_assembler.hpp"
#include "megavit/rope.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

std::vector<megavit::DecodedFrame> MakeFrames() {
    std::vector<megavit::DecodedFrame> frames;
    for (int t = 0; t < 8; ++t) {
        megavit::DecodedFrame frame;
        frame.frame_index = t;
        frame.timestamp_seconds = 0.25 * t;
        frame.width = 64;
        frame.height = 64;
        frame.key_frame = (t % 4) == 0;
        frame.rgb.resize(64 * 64 * 3);
        for (int y = 0; y < 64; ++y) {
            for (int x = 0; x < 64; ++x) {
                const size_t at = (static_cast<size_t>(y) * 64 + x) * 3;
                frame.rgb[at] = static_cast<uint8_t>(x + t);
                frame.rgb[at + 1] = static_cast<uint8_t>(y + t * 2);
                frame.rgb[at + 2] = static_cast<uint8_t>(x + y + t * 3);
            }
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}

}  // namespace

int main() {
    megavit::CodecFrontendConfig config;
    config.canvas_width = 64;
    config.canvas_height = 64;
    config.group_size = 4;
    config.images_per_group = 1;
    config.target_canvases = 2;
    config.min_group_frames = 4;
    config.validate();

    megavit::CodecFrontend frontend(config);
    const auto canvases = frontend.Build(MakeFrames());
    assert(canvases.size() == 2);
    for (const auto& canvas : canvases) {
        assert(canvas.rgb.size() == 64 * 64 * 3);
        assert(canvas.patch_positions.size() == 16);
        assert(canvas.patch_timestamps.size() == 16);
        for (size_t i = 0; i < canvas.patch_positions.size(); i += 4) {
            const auto& tl = canvas.patch_positions[i];
            const auto& tr = canvas.patch_positions[i + 1];
            const auto& bl = canvas.patch_positions[i + 2];
            const auto& br = canvas.patch_positions[i + 3];
            assert(tl.t == tr.t && tl.t == bl.t && tl.t == br.t);
            assert(tr.h == tl.h && tr.w == tl.w + 1);
            assert(bl.h == tl.h + 1 && bl.w == tl.w);
            assert(br.h == tl.h + 1 && br.w == tl.w + 1);
        }
    }

    const auto rope = megavit::BuildMageVisionRope(canvases[0].patch_positions);
    assert(rope.head_dim == 64);
    assert(rope.sin.size() == 16 * 64);
    assert(rope.cos.size() == 16 * 64);
    for (float value : rope.sin) assert(std::isfinite(value));
    for (float value : rope.cos) assert(std::isfinite(value));

    const auto prompt = megavit::BuildPromptPlan(canvases);
    assert(prompt.total_visual_tokens == 8);
    assert(!prompt.spans.empty());
    int expected_offset = 0;
    for (const auto& span : prompt.spans) {
        assert(span.embedding_offset == expected_offset);
        expected_offset += span.token_count;
    }
    assert(expected_offset == prompt.total_visual_tokens);
    assert(prompt.vision_content.find(" seconds><|vision_start|>") != std::string::npos);
    assert(prompt.vision_content.find("<|vision_end|>") != std::string::npos);

    std::cout << "Mage-Vit frontend tests passed\n";
    return 0;
}

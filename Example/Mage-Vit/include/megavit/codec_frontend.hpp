#pragma once

#include "megavit/types.hpp"

#include <string>
#include <vector>

namespace megavit {

struct CodecFrontendConfig {
    int patch_size = 16;
    int spatial_merge_size = 2;
    // Fixed landscape profile emitted by codec-video-prep for the reference
    // sample: 576 patches and 144 merged visual tokens per canvas.
    int canvas_width = 512;
    int canvas_height = 288;
    int group_size = 32;
    int images_per_group = 4;
    int target_canvases = 32;
    int min_group_frames = 8;
    // Maximum fraction of a group's selected blocks supplied by one frame.
    // A second uncapped pass fills the budget if this cap is too restrictive.
    float per_frame_cap_ratio = 0.25f;
    // Makes anchor-frame blocks survive the top-K stage. This is deliberately
    // a boost rather than an unconditional keep because the fixed canvas has a
    // hard capacity.
    float keyframe_score_bonus = 1000000.0f;
    bool require_full_canvases = true;

    int block_pixels() const { return patch_size * spatial_merge_size; }
    int canvas_patch_count() const {
        return (canvas_width / patch_size) * (canvas_height / patch_size);
    }
    int canvas_token_count() const {
        return canvas_patch_count() /
               (spatial_merge_size * spatial_merge_size);
    }
    int blocks_per_canvas() const {
        return (canvas_width / block_pixels()) *
               (canvas_height / block_pixels());
    }
    int sampled_frame_budget() const {
        return (target_canvases / images_per_group) * group_size;
    }

    void validate() const;
};

class CodecFrontend {
public:
    explicit CodecFrontend(CodecFrontendConfig config);

    // The decoder owns demux/decode. This class owns uniform sampling,
    // codec-score aggregation, top-K selection, canvas packing and Mage 3D
    // patch-position generation.
    std::vector<Canvas> Build(const std::vector<DecodedFrame>& decoded_frames) const;

    const CodecFrontendConfig& config() const { return config_; }

private:
    CodecFrontendConfig config_;
};

// Evaluation bundle: PPM canvases plus a JSON manifest containing the exact
// patch positions consumed by Mage-ViT. No JPEG round-trip is used at runtime.
void WriteFrontendBundle(
    const std::string& output_directory,
    const CodecFrontendConfig& config,
    const std::vector<Canvas>& canvases);

}  // namespace megavit

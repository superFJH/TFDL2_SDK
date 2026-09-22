#pragma once
#include "tfllm/engine.h"

namespace tfllm {
struct VisionConfig {
    std::string projector="qwen3vl_merger";
    size_t layers=0,hidden=0,intermediate=0,heads=0,output=0,patch=0,merge=2,position_side=0;
    size_t temporal=2,window=0,full_attention_period=0;
    bool rms=false;
    std::string activation="gelu";
    std::vector<size_t> feature_layers;
    float epsilon=1e-6f;
    std::array<float,3> mean{},stddev{};
    std::vector<size_t> deepstack;
    void Validate() const;
};
struct VisionModel {
    VisionConfig config;
    std::string identity;
    std::map<std::string,Tensor> tensors; // Canonical GGUF names; patch is flattened [N,C*T*H*W].
    void Validate() const;
};
// GGUF codec supplied by the llama adapter. No vision execution uses llama.
std::shared_ptr<VisionModel> LoadVisionGguf(const std::string& file);
class VisionEncoder {
public:
    VisionEncoder(std::shared_ptr<const VisionModel>,std::shared_ptr<Linear>);
    void Prepare(size_t max_rows=1024);
    // Normalized FP16 CHW image, already resized to multiples of patch*merge.
    // Temporal Conv3D uses two identical frames for a still image.
    std::shared_ptr<VisualTokens> Encode(const Activation&,size_t height,size_t width,const std::string& identity) const;
    // Qwen3-VL: two distinct normalized frames in TCHW order. Each pair is
    // one independent spatial attention group, matching HF's cu_seqlens.
    std::shared_ptr<VisualTokens> EncodeTemporalPair(const Activation&,size_t height,size_t width,const std::string& identity) const;
    const VisionConfig& Config() const {return model_->config;}
private:
    std::shared_ptr<VisualTokens> EncodeQwen3(const Activation&,size_t,size_t,size_t frames,const std::string&) const;
    std::shared_ptr<VisualTokens> EncodeDense(const Activation&,size_t,size_t,const std::string&) const;
    std::shared_ptr<const VisionModel> model_;
    std::shared_ptr<Linear> linear_;
    std::map<std::string,std::vector<float>> vectors_;
    std::map<std::string,Activation> half_vectors_;
};
void ValidateDenseVision(const VisionModel&);
}

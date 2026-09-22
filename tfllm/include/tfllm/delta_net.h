#pragma once
#include "tfllm/engine.h"

namespace tfllm {
// HF repeats each K head; the pinned GGUF converter tiles K heads and
// reorders all V-dependent weights. Never mix weights from the two layouts.
enum class DeltaHeadOrder { Grouped, Tiled };
struct DeltaGeometry {
    size_t key_heads=0, value_heads=0, key_dim=0, value_dim=0, kernel=4;
    float epsilon=1e-6f;
    DeltaHeadOrder order=DeltaHeadOrder::Tiled;
    void Validate() const;
    size_t KeyWidth() const;
    size_t ValueWidth() const;
    size_t ConvWidth() const;
};
DeltaGeometry DeltaConfig(const Config&);
struct DeltaState {
    size_t position=0;
    // History [channels, kernel-1], oldest first; matrix [V heads, K dim, V dim].
    // This is an explicit FP32 ABI, not a view of llama's private state layout.
    std::vector<float> history, matrix;
    explicit DeltaState(const DeltaGeometry&);
    void Reset();
    void Validate(const DeltaGeometry&) const;
};
struct DeltaWeights {
    // GGUF convention: a is already -exp(HF A_log); norm is the direct scale.
    std::vector<float> convolution, a, dt_bias, norm;
    void Validate(const DeltaGeometry&) const;
};
// Projected inputs are row-major [tokens, channels]. No padding tokens.
// Returns normalized, SiLU(z)-gated [tokens, V width], before out projection.
// Strong state guarantee: exceptions leave the supplied state unchanged.
std::vector<float> RunDeltaNetFp32(const DeltaGeometry&,const DeltaWeights&,
    DeltaState&,size_t tokens,const std::vector<float>& qkv,const std::vector<float>& z,
    const std::vector<float>& alpha,const std::vector<float>& beta);

// A complete residual DeltaNet + gated FFN block. All eight dense projections
// use the supplied Linear (CPU or NPU40T); the recurrent core stays CPU FP32.
// Weights use native/GGUF names, with post_attention_norm mapped to ffn_norm.
// State belongs to the caller/sequence and commits only after the FFN succeeds.
Activation RunDeltaNetBlock(Linear&,const Model&,const std::string& prefix,
    const DeltaGeometry&,DeltaState&,const Activation& hidden);
}

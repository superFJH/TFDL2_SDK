#pragma once
#include "tfllm/engine.h"

namespace tfllm {
class VisionEncoder;
struct RgbImage {size_t height=0,width=0;std::vector<uint8_t> pixels;};
class LlamaVision {
public:
    ~LlamaVision();
    std::pair<std::vector<int32_t>,std::shared_ptr<PromptInput>> Prompt(const std::string&,const std::vector<RgbImage>&,bool add_special=false);
private:
    friend class LlamaModel;
    LlamaVision(const std::string&,const void*,std::shared_ptr<void>,std::shared_ptr<VisionEncoder>,const Config&,std::array<int32_t,3>,size_t);
    struct Impl;std::unique_ptr<Impl> impl_;
};
struct LlamaComponents {
    std::shared_ptr<KvCache> cache;
    std::shared_ptr<Backend> decode;
    std::shared_ptr<CacheBridge> bridge;
};
// Immutable llama weights/vocabulary shared by FP16 KV contexts.
// Load and validate once per engine, not once per conversation.
class LlamaModel {
public:
    LlamaModel(const std::string& gguf, const Config& config);
    LlamaComponents CreateContext(size_t capacity, int threads = 4) const;
    // One context, one persistent CPU pool, separate fixed KV streams. Bind
    // each returned component to exactly one Session. Ready short/decode jobs
    // are combined at step boundaries; native prefill writes its own stream.
    // threads is the TOTAL decode budget, independent of sequences. Waiting
    // to coalesce a batch is bounded; no request waits for a slow peer.
    // Experimental qwen35 uses private contexts per sequence for recurrent
    // cells; threads is per context and decode jobs are not coalesced yet.
    std::vector<LlamaComponents> CreateBatchContexts(size_t capacity, size_t sequences,
                                                    int threads = 4, unsigned batch_wait_us = 200) const;
    std::vector<int32_t> Tokenize(const std::string& text, bool add_special = true) const;
    std::string Piece(int32_t token, bool special = false) const;
    bool IsEnd(int32_t token) const;
    std::string ChatTemplate() const;
    std::array<int32_t,3> MropeSections() const;
    size_t DeepstackLayers() const;
    bool IsLocateAnything() const;
    std::pair<std::vector<int32_t>,std::shared_ptr<PromptInput>> ImagePrompt(
        const std::vector<int32_t>& tokens,const std::vector<std::shared_ptr<const VisualTokens>>& images) const;
    // Ordered image / video-pair spans; timestamp text is already tokenized.
    std::pair<std::vector<int32_t>,std::shared_ptr<PromptInput>> QwenMediaPrompt(
        const std::vector<int32_t>& tokens,const std::vector<std::shared_ptr<const VisualTokens>>& features,
        const std::vector<bool>& video_pairs) const;
    // LocateAnything slow/AR: scalar positions and causal image spans.
    std::pair<std::vector<int32_t>,std::shared_ptr<PromptInput>> LocateImagePrompt(
        const std::vector<int32_t>& tokens,const std::vector<std::shared_ptr<const VisualTokens>>& images) const;
    std::unique_ptr<LlamaVision> CreateVision(const std::string& mmproj,std::shared_ptr<VisionEncoder>,size_t max_image_tokens) const;
    std::string Bos() const;
    std::string Eos() const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
// Creates an exclusively owned CPU llama context, sequence 0, FP16 KV, no
// cache shifting, full-capacity SWA, no MLA. Returned KV views alias its actual cache buffers.
// Separate calls create independent sessions; the caller never receives a raw
// llama context that could bypass Session's execution lock.
LlamaComponents CreateLlama(const std::string& gguf, const Config& config,
                            size_t capacity, int threads = 4);
// Stable content identity of the source GGUF. Used by conversion and binding
// to reject a same-shaped but different model. Not a cryptographic signature.
std::string GgufIdentity(const std::string& path);
// Streams supported dense models into native packages. Others produce an
// explicit llama fallback manifest referencing the unchanged source GGUF.
// If decode_q4_if_bf16 is set, floating BF16 input additionally produces a
// paired Q4_0 GGUF at that new path. Other inputs leave that path untouched.
// NPU tensors always come directly from source. Both identities are recorded;
// LlamaModel accepts either the original source or this exact paired GGUF.
Config ConvertGguf(const std::string& source,const std::string& prefix,
                   bool f32=false,bool require_native=false,
                   const std::string& decode_q4_if_bf16={});
class GgufSession {
public:
    GgufSession(const std::string& gguf,size_t capacity,int threads=4,
                const std::string& expected_identity={});
    ~GgufSession();
    std::vector<float> Append(const std::vector<int32_t>& tokens);
    std::vector<float> Prefill(const std::vector<int32_t>& prompt);
    std::vector<int32_t> Tokens() const;
    GgufSession(const GgufSession&)=delete;
    GgufSession& operator=(const GgufSession&)=delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

#pragma once
#include <stdexcept>
#include "tfllm/kv_cache.h"
#include "tfllm/activation.h"
#include "tfllm/linear_view.h"
#include "tfllm/multimodal.h"
#include <mutex>

namespace tfllm {
enum class Route { Prefill, Short };
struct Execution {
    size_t past = 0, length = 0, query_bucket = 0, key_bucket = 0;
    Route route = Route::Prefill;
    std::shared_ptr<const PromptInput> input;
    size_t VisibleKeys(size_t query_row) const;
};
struct Policy {
    size_t short_threshold = 32, max_prefill = 1024, alignment = 64;
    Execution Next(size_t past, size_t remaining, size_t capacity) const;
};
class Backend {
public:
    virtual ~Backend() = default;
    // Returns last-real-token logits. Complete KV writes before returning or
    // throwing. Pending DMA may be quarantined only in private buffers that
    // cannot touch this KV. External engine cell metadata uses CacheBridge.
    virtual std::vector<float> Run(const Model&, KvCache&, const Execution&,
                                   const int32_t* tokens) = 0;
    virtual const char* Name() const = 0;
    virtual void Prepare(const Model&,size_t /*max_rows*/) {}
};
// Metadata and visibility handoff for a cache shared with another engine.
class CacheBridge {
public:
    virtual ~CacheBridge() = default;
    // Validate/synchronize the handoff without changing committed metadata.
    virtual void Begin(size_t past, size_t length, Route route) = 0;
    virtual void Commit(size_t past, const std::vector<int32_t>& tokens, Route route) = 0;
    virtual void Truncate(size_t length) = 0;
    virtual void SetInput(std::shared_ptr<const PromptInput> input) {if(input)throw std::invalid_argument("bridge does not support visual inputs");}
};
class Session {
public:
    Session(std::shared_ptr<const Model> model, std::shared_ptr<KvCache> cache,
            std::shared_ptr<Backend> prefill, std::shared_ptr<Backend> short_backend,
            Policy policy = {}, std::shared_ptr<CacheBridge> bridge = {});
    std::vector<float> Append(const std::vector<int32_t>& tokens);
    // Reuse the longest identical prefix. A divergent suffix is invalidated.
    std::vector<float> Prefill(const std::vector<int32_t>& complete_prompt);
    std::vector<float> Prefill(const std::vector<int32_t>& complete_prompt,std::shared_ptr<const PromptInput> input);
    struct PrefixMatch {size_t stored=0, common=0;};
    // Read-only slot selection; uses the same image/position checks as Prefill.
    // Replaying a shorter prompt can still require its final logits to be recomputed.
    PrefixMatch MatchPrefix(const std::vector<int32_t>& prompt,const PromptInput* input=nullptr) const;
    void Truncate(size_t length);
    std::vector<int32_t> Tokens() const;
    std::vector<Execution> LastExecutions() const;
private:
    std::vector<float> AppendLocked(const std::vector<int32_t>& tokens);
    size_t CommonPrefixLocked(const std::vector<int32_t>& prompt,const PromptInput* input) const;
    void TruncateLocked(size_t length);
    std::shared_ptr<const Model> model_;
    std::shared_ptr<KvCache> cache_;
    std::shared_ptr<Backend> prefill_, short_;
    std::shared_ptr<CacheBridge> bridge_;
    Policy policy_;
    mutable std::mutex mutex_;
    std::vector<int32_t> tokens_;
    std::vector<float> logits_;
    std::vector<Execution> executions_;
    bool failed_ = false;
    std::shared_ptr<const PromptInput> input_;
};
// A linear implementation may use NPU40T. Input/output are [rows, channels];
// immutable weights are [output channels, input channels].
class Linear {
public:
    virtual ~Linear() = default;
    // False after an unrecoverable device failure. A pool retires this worker
    // and never replays the failed operation on another device.
    virtual bool Healthy() const noexcept { return true; }
    virtual void Prepare(const Tensor&,size_t /*rows*/) {}
    // Release cached static packing. Call after the owner's last invocation;
    // in-flight backend jobs retain their own references.
    virtual void Forget(const Tensor&) {}
    // Snapshot a dynamic operand for all query tile sizes before entering the
    // execution workspace. Backends may quantize/pack into ordinary host RAM.
    // Recreate after any KV change; the returned operand is immutable.
    virtual Tensor PrepareDynamic(const Tensor& weight,const std::vector<size_t>& /*rows*/) {return weight;}
    // Query rows per noncausal vision Attention task. Backends choose a
    // bounded score working set; each completed task remains a scheduling
    // boundary. This does not change which keys are visible to a query.
    virtual size_t AttentionTileRows(size_t keys) const;
    // Capability for physically padded K/V dimensions. Allows callers to
    // avoid constructing padded snapshots when the fused path is unavailable.
    virtual bool SupportsAttentionFp16(size_t /*keys*/,size_t /*head_dim*/) const {return false;}
    virtual bool RunAttentionFp16(const Tensor&,const Tensor&,Fp16InputView,
                                 Fp16OutputView,const AttentionMask&) {return false;}
    virtual std::vector<float> Run(const Tensor& weight, const std::vector<float>& input,
                                   size_t rows) = 0;
    // Production prefill uses binary16 end to end. The defaults are explicit
    // compatibility bridges for reference/third-party implementations only.
    virtual Activation RunFp16(const Tensor&,const Activation&,size_t rows);
    // A shorter logical K is zero-padded; only output.columns are written.
    // Defaults preserve compatibility with existing CPU/third-party backends.
    virtual void RunFp16Into(const Tensor&,Fp16InputView,Fp16OutputView);
    virtual void AccumulateFp16View(const Tensor&,Fp16InputView,float* sum,size_t stride);
    virtual bool SupportsQuantizedInput(size_t /*columns*/) const {return false;}
    virtual void RunQuantizedInto(const Tensor&,const QuantizedInput&,Fp16OutputView);
    // Only backends supporting simultaneous calls opt into head pipelining.
    virtual size_t VisionAttentionConcurrency() const {return 1;}
    // Internal long-K reduction: add unrounded FP32 partials into operator
    // scratch and narrow once after the full K. Not an activation boundary.
    virtual void AccumulateFp16(const Tensor&,const Activation&,size_t rows,float* sum,size_t stride);
};
// Adapt dense weights to finite hardware dimensions. Each K partial is
// dequantized by the inner backend, then accumulated in FP32. Padding is zero.
struct LinearLimits {
    size_t k=16384, n=16384, alignment=64;
    size_t cached_weight_bytes=256ull*1024*1024, cached_weights=32;
};
std::shared_ptr<Linear> CreateBlockedLinear(std::shared_ptr<Linear> inner,LinearLimits limits={});
class Attention {
public:
    virtual ~Attention() = default;
    virtual std::vector<float> Run(const Config&, KvCache&, size_t layer,
        const Execution&, const std::vector<float>& query) = 0;
    virtual Activation RunFp16(const Config&,KvCache&,size_t layer,const Execution&,const Activation&);
};
std::shared_ptr<Attention> CreateTiledAttention(std::shared_ptr<Linear> linear);
// Native FP16 prefill, SIMD CPU operators and shared persistent CPU workers.
// Without a Linear, use FP32 dot accumulation with FP16 operator boundaries.
std::shared_ptr<Backend> CreateFp16Prefill(std::shared_ptr<Linear> linear = {});
std::shared_ptr<Linear> CreateCpuFp16Linear();
class ReferenceBackend : public Backend {
public:
    explicit ReferenceBackend(std::shared_ptr<Linear> linear = {}, std::shared_ptr<Attention> attention = {});
    std::vector<float> Run(const Model&, KvCache&, const Execution&, const int32_t*) override;
    const char* Name() const override { return linear_ ? "dense-decoder-accelerated" : "dense-decoder-reference"; }
private:
    std::shared_ptr<Linear> linear_;
    std::shared_ptr<Attention> attention_;
};
}

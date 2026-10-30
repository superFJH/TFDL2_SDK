#pragma once
#include "tfllm/engine.h"
#include "tfllm/scheduler.h"
#include "tfllm/spatial_conv.h"
namespace tfllm {
class AudioConvolution;
struct NpuOptions {
    int chip=0;
    // Fixed pools use TFLLM_NPU_RUNTIME_MIB for the per-window split.
    double gpu_memory_utilization=0.8; // floor(registered HugePage GiB * fraction).
    bool elastic_memory=false; // Torch opts in; native serve retains fixed reservation.
    size_t max_plans_per_pair=0; // 0: no CPU-template count cap; otherwise chip template limit = value * 4.
    // Explicit Warmup() compiles these M values for prepared static geometries.
    // The serve frontend selects all 16-row sizes; SDK callers opt in.
    std::vector<size_t> jit_warmup_rows;
    // Model-specific disk directory for immutable CPU recipes. Empty disables;
    // serve supplies a directory under --cache-dir. No physical buffers persist.
    std::string jit_cache_dir;
    size_t queue_capacity=64;
    int timeout_ms=30000, lock_timeout_ms=30000;
    // Empty: TFLLM_NPU_DUMP_DIR or ./tfllm_npu_dumps. "0": disabled.
    // Fault-only, private files; immutable commands/geometry, never activations.
    std::string failure_dump_dir;
    std::vector<int> worker_cpus; // Four persistent pair workers.
    bool verify_outputs=false;
    // Treat all eight TFACCs as equal compute capacity. Workload gates are
    // tunable, not calibrated timings; any idle pair may execute a shard.
    bool adaptive_split=true;
    bool trace_split=false; // Diagnostic geometry/selection logs; default off.
    size_t max_compute_cores=8; // 1,2,4,8 per GEMM/Conv image/attention head; leases unchanged.
    size_t split_min_macs=4*1024*1024;
    bool use_cache=true; // NPU hardware cache; TFLLM_USE_CACHE=0 bypasses it.
    bool cache_in_command=true; // PublishOutputs; large static Linear may use external publication.
    bool cache_prepare_async=true; // Reuse command SyncAll instead of CPU ACK polling.
    bool cache_writeback_async=false; // Board-verified synchronous release by default.
    bool cache_writeback_range=false; // Experimental range scope, separate from ACK timing.
    bool cache_writeback_check=false; // Compare range output against a full flush, on board.
    // Active QK INT32 budget across up to four independent pair lanes, per slot.
    // Pipeline uses two slots. Reduce lanes if 16 rows/lane cannot fit.
    // This is not the total execution workspace.
    size_t attention_score_bytes=4*1024*1024;
    // Extra executed query rows relative to RoundUp(total query rows,16).
    // Counts all lane/tile tails, not just the live score buffers. 0..100.
    unsigned attention_padding_percent=25;
    size_t attention_lease_tiles=4; // Fairness boundary; release after this many tiles/pair.
    bool attention_fused=true,attention_pipeline=true;
    // Opt-in U8 centered scores: restore (code-128)*2^shift. 8..16;
    // zero retains I32. Saturated rows retry on NPU in groups of 16.
    // Unsaturated rows still incur rounding loss; not a general LLM accuracy policy.
    unsigned attention_qk_shift=0;
    bool attention_fast=false; // Opt-in common FAST attention, logical head dimensions <=256.
    int attention_fast_override=-1; // -1 reads environment; 0/1 is an explicit API choice.
    bool attention_qk_overlap=true; // Fuse the preceding score tile during the next QK.
    bool attention_workspace_reuse=true; // Private, pressure-evictable encode scope.
    bool workspace_reuse=true; // Static/dynamic GEMM, per prefill/encode caller.
    size_t workspace_cache_bytes=1024ull*1024*1024; // Idle A/C/scratch + commands, bounded with count below.
    size_t workspace_cache_instances=32; // Independent cap; deep/concurrent CNNs can explicitly raise it.
    // 1..32 CPU participants per lane, including the pair worker. Also grows
    // the shared CPU pool to at least this value minus one background threads.
    // Ordinary CPU kernels retain their per-call limit of eight participants.
    size_t attention_fuse_workers=16;
    size_t linear_tile_rows=0; // Ordinary Linear override; 0 keeps narrow/wide defaults.
    bool conv_workspace_reuse=false; // Bounded idle cache for repeated Torch Conv.
    size_t window_batch=0; // Windows per core/command chain; Torch opts in.
    bool forward_cache=false; // Torch opts in; native serving retains request-scoped storage.
    size_t cache_idle_ms=60000; // Idle execution/template expiry; 0 disables timed expiry.
    size_t template_cache_bytes=64ull*1024*1024; // Forward-cache CPU recipe budget, <=512 entries.
};
// Strict integer overrides, read when constructing a scheduler. Unset keeps
// explicit SDK options; environment values take precedence when present.
// SCORE_MIB: 1..16 MiB per slot; FUSE_WORKERS: 1..32 participants per lane;
// PADDING_PERCENT: 0..100 extra percent over minimum 16-row query alignment;
// QK_OVERLAP: 0/1, default 1; QK_SHIFT: 0 (I32 default) or 8..16 (lossy U8).
// U8 requires the matching SDK, padded head dim <=256, and keeps I32 retry storage.
// WORKSPACE_REUSE: 0/1. These names have the TFLLM_NPU_ATTENTION_ prefix.
// TFLLM_NPU_WORKSPACE_REUSE: 0/1; TFLLM_NPU_WORKSPACE_CACHE_MIB: 1..4096.
// TFLLM_NPU_WORKSPACE_CACHE_INSTANCES: 1..4096, default 32. The byte budget,
// idle timeout and allocation-pressure eviction remain independent limits.
// TFLLM_NPU_ELASTIC_MEMORY: 0/1 (Torch default 1, native serve default 0).
// TFLLM_NPU_FORWARD_CACHE: 0/1 (Torch default 1); CACHE_IDLE_MS: 0..86400000;
// TEMPLATE_CACHE_MIB: 1..4096. The latter two also have the TFLLM_NPU_ prefix.
// TFLLM_NPU_JIT_ROWS: all (16..1024 step 16), 0, or comma-separated 1..1024.
// Only Warmup() uses these rows; execution keeps its existing 16-row alignment.
// TFLLM_USE_CACHE: 0/1, default 1; independent of command/workspace reuse.
// TFLLM_NPU_ADAPTIVE_SPLIT: 0/1; 0 restores the SDK Grid without workload gates.
// TFLLM_NPU_MAX_CORES: 1/2/4/8, compute cap for all operator types.
// TFLLM_NPU_SPLIT_MIN_MACS: 1..1073741824, default 4194304.
// TFLLM_NPU_TRACE_SPLIT: 0/1, print selected shard/lane geometry.
// TFLLM_NPU_CACHE_PREPARE_ASYNC: 0/1, default 1; 0 restores CPU prepare ACK waits.
// TFLLM_NPU_LINEAR_CACHE_POLICY: auto (default), command or external. Static
// INT32 Linear only; auto publishes externally at >=1.5 MiB output/core.
// Attention/Conv retain cache_in_command; CACHE_IN_COMMAND=0 takes precedence.
// TFLLM_NPU_LINEAR_LAYOUT: xw (default), wxt, wxt15, wxt20 or auto.
// TFLLM_NPU_LINEAR_AUTOTUNE: 0/1, default 0; optional cold static-Linear search.
// Search compares layouts, 1/2/4/6/8 cores, original grids and cross-aligned
// full/lite partitions. Complete-call timing selects the winner; I32 output
// and the existing quantization contract are preserved.
// TFLLM_NPU_LINEAR_TUNE_ITERS: 1..31 timed calls/candidate, default 5.
// TFLLM_NPU_LINEAR_TUNE_CONCURRENCY: 1 (latency), 2 or 4 concurrent callers.
// TFLLM_NPU_CACHE_WRITEBACK_ASYNC: 0/1, default 0; 1 waits at the CPU consumer.
// TFLLM_NPU_CACHE_WRITEBACK_RANGE: 0/1, default 0; requires ASYNC=1.
// TFLLM_NPU_CACHE_WRITEBACK_CHECK: 0/1, default 0; requires cached ASYNC+RANGE.
NpuOptions NpuOptionsFromEnvironment(NpuOptions options={});
// Also useful for deployment planning without opening an NPU.
uint64_t NpuMemoryBudget(uint64_t hugepage_bytes,double utilization);
struct NpuMemoryStats {uint64_t registered_bytes=0,reserved_bytes=0,used_bytes=0,budget_bytes=0;};
struct NpuJitShape {size_t rows=0,k=0,n=0,cores=8;};
class NpuScheduler final : public Linear {
public:
    explicit NpuScheduler(NpuOptions options={});
    ~NpuScheduler();
    std::vector<float> Run(const Tensor&,const std::vector<float>&,size_t) override;
    Activation RunFp16(const Tensor&,const Activation&,size_t) override;
    void RunFp16Into(const Tensor&,Fp16InputView,Fp16OutputView) override;
    void RunFp32BridgeInto(const Tensor&,Fp32InputView,Fp32OutputView) override;
    void RunFp32Into(const Tensor&,Fp32InputView,Fp32OutputView) override;
    size_t LinearTileRows(size_t k,size_t n) const override;
    bool SupportsQuantizedInput(size_t columns) const override;
    size_t VisionAttentionConcurrency() const override;
    void RunQuantizedInto(const Tensor&,const QuantizedInput&,Fp16OutputView) override;
    void RunQuantizedFp32Into(const Tensor&,const QuantizedInput&,Fp32OutputView) override;
    bool Healthy() const noexcept override;
    std::unique_ptr<LinearWorkspaceScope> BeginWorkspaceReuse() override;
    void Prepare(const Tensor&,size_t rows) override;
    void Forget(const Tensor&) override;
    std::shared_ptr<SpatialConvWeight> PrepareSpatial(const SpatialConvSpec&,const uint16_t*);
    std::shared_ptr<SpatialConvWeight> PrepareSpatial(const std::shared_ptr<SpatialConvWeight>&);
    // Pixel-symmetric 1x1 weights; bias belongs after output dequantization.
    std::shared_ptr<SpatialConvWeight> PreparePointwise(const SpatialConvSpec&,const uint16_t*);
    bool SupportsPointwiseNative(size_t input_channels,size_t output_channels,size_t height,size_t width) const;
    void WarmupSpatial(const std::shared_ptr<SpatialConvWeight>&,size_t height,size_t width);
    void RunSpatial(const std::shared_ptr<SpatialConvWeight>&,const uint16_t*,uint16_t*,size_t batch,size_t height,size_t width,
                    const uint16_t* bias=nullptr);
    void RunPointwiseFp32(const std::shared_ptr<SpatialConvWeight>&,const float*,float*,size_t batch,size_t height,size_t width);
    void RunSpatialFp32(const std::shared_ptr<SpatialConvWeight>&,const float*,float*,size_t batch,size_t height,size_t width);
    Tensor PrepareDynamic(const Tensor&,const std::vector<size_t>& rows) override;
    size_t AttentionTileRows(size_t keys) const override;
    bool SupportsAttentionFp16(size_t keys,size_t head_dim) const override;
    bool AttentionFastEnabled() const noexcept override;
    bool RunAttentionFp16(const Tensor&,const Tensor&,Fp16InputView,
                         Fp16OutputView,const AttentionMask&) override;
    bool RunWindowAttentionFp16(const uint16_t*,const uint16_t*,const uint16_t*,uint16_t*,
                               size_t jobs,size_t queries,size_t dim,float scale,
                               size_t keys=64,size_t heads=1,const size_t* strides=nullptr) override;
    bool RunWindowAttentionFp32(const float*,const float*,const float*,float*,
                               size_t jobs,size_t queries,size_t dim,size_t heads,float scale,const size_t* strides) override;
    bool RunWindowAttentionFp32Native(const float*,const float*,const float*,float*,
                               size_t jobs,size_t queries,size_t dim,size_t heads,float scale,const size_t* strides,
                               size_t keys=64) override;
    bool RunAttentionFp32Scoped(const Tensor&,const Tensor&,Fp32InputView,Fp32OutputView,
                      const AttentionMask&,const std::shared_ptr<AttentionWorkspaceScope>&) override;
    std::shared_ptr<AttentionWorkspaceScope> CreateAttentionWorkspaceScope() override;
    bool RunAttentionFp16Scoped(const Tensor&,const Tensor&,Fp16InputView,Fp16OutputView,
                               const AttentionMask&,const std::shared_ptr<AttentionWorkspaceScope>&) override;
    void PrepareModel(const Model&,size_t max_rows=1024);
    // CPU recipes only: no inference, execution buffers, KV changes or leases.
    // Extra shapes may include dynamic QK/AV (cores=2) captured from a profile.
    void Warmup(const std::vector<NpuJitShape>& extra_shapes={});
    PoolStats Stats() const;
    NpuMemoryStats MemoryStats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Adaptive GEMMs use a workload-based 1/2/4/8 shard cap and uniform capacity.
// Tasks migrate freely between all four pairs using matching command variants;
// adaptive_split=false restores the SDK Grid without workload gates.
// One request can occupy four pairs; two competing groups get two each, four
// get one each. Rebalance only at completed task boundaries; excess jobs wait.
// CPU templates hold no DMA allocations. Initial reservation and direct command
// cache growth transiently take all process leases. Cached command suballocation
// needs no device lease; execution takes its pair only. Prefill/encode scopes
// may retain pressure-evictable private workspaces; idle engines hold no
// execution instances or driver locks. Every in-flight instance has independent
// mutable commands, A/C and temporary K/V workspace.
std::shared_ptr<Linear> CreateNpuLinear(NpuOptions options={});
std::shared_ptr<Backend> CreateNpuPrefill(NpuOptions options={});
using NpuPoolOptions=NpuOptions;
struct NpuPrefillPool {
    std::shared_ptr<Backend> backend;
    std::shared_ptr<NpuScheduler> scheduler;
};
NpuPrefillPool CreateNpuPrefillPool(NpuPoolOptions options={});
std::shared_ptr<AudioConvolution> CreateNpuAudioConvolution(std::shared_ptr<NpuScheduler>);
}

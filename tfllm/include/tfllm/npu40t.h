#pragma once
#include "tfllm/engine.h"
#include "tfllm/scheduler.h"
#include "tfllm/spatial_conv.h"
namespace tfllm {
class AudioConvolution;
struct NpuOptions {
    int chip=0;
    // Per-window split is independently configured by TFLLM_NPU_RUNTIME_MIB.
    double gpu_memory_utilization=0.8; // floor(registered HugePage GiB * fraction).
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
    bool use_cache=true; // NPU hardware cache; TFLLM_USE_CACHE=0 bypasses it.
    bool cache_in_command=true; // Opt-in PublishOutputs template policy; board validation required.
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
    bool attention_workspace_reuse=true; // Private, pressure-evictable encode scope.
    bool workspace_reuse=true; // Static/dynamic GEMM, per prefill/encode caller.
    size_t workspace_cache_bytes=1024ull*1024*1024; // Idle A/C/scratch + commands; <=32 instances.
    // 1..32 CPU participants per lane, including the pair worker. Also grows
    // the shared CPU pool to at least this value minus one background threads.
    // Ordinary CPU kernels retain their per-call limit of eight participants.
    size_t attention_fuse_workers=16;
};
// Strict integer overrides, read when constructing a scheduler. Unset keeps
// explicit SDK options; environment values take precedence when present.
// SCORE_MIB: 1..16 MiB per slot; FUSE_WORKERS: 1..32 participants per lane;
// PADDING_PERCENT: 0..100 extra percent over minimum 16-row query alignment;
// WORKSPACE_REUSE: 0/1. These names have the TFLLM_NPU_ATTENTION_ prefix.
// TFLLM_NPU_WORKSPACE_REUSE: 0/1; TFLLM_NPU_WORKSPACE_CACHE_MIB: 1..4096.
// TFLLM_NPU_JIT_ROWS: all (16..1024 step 16), 0, or comma-separated 1..1024.
// Only Warmup() uses these rows; execution keeps its existing 16-row alignment.
// TFLLM_USE_CACHE: 0/1, default 1; independent of command/workspace reuse.
// TFLLM_NPU_CACHE_PREPARE_ASYNC: 0/1, default 1; 0 restores CPU prepare ACK waits.
// TFLLM_NPU_CACHE_WRITEBACK_ASYNC: 0/1, default 0; 1 waits at the CPU consumer.
// TFLLM_NPU_CACHE_WRITEBACK_RANGE: 0/1, default 0; requires ASYNC=1.
// TFLLM_NPU_CACHE_WRITEBACK_CHECK: 0/1, default 0; requires cached ASYNC+RANGE.
NpuOptions NpuOptionsFromEnvironment(NpuOptions options={});
// Also useful for deployment planning without opening an NPU.
uint64_t NpuMemoryBudget(uint64_t hugepage_bytes,double utilization);
struct NpuMemoryStats {uint64_t registered_bytes=0,reserved_bytes=0,used_bytes=0;};
struct NpuJitShape {size_t rows=0,k=0,n=0,cores=8;};
class NpuScheduler final : public Linear {
public:
    explicit NpuScheduler(NpuOptions options={});
    ~NpuScheduler();
    std::vector<float> Run(const Tensor&,const std::vector<float>&,size_t) override;
    Activation RunFp16(const Tensor&,const Activation&,size_t) override;
    void RunFp16Into(const Tensor&,Fp16InputView,Fp16OutputView) override;
    bool SupportsQuantizedInput(size_t columns) const override;
    size_t VisionAttentionConcurrency() const override;
    void RunQuantizedInto(const Tensor&,const QuantizedInput&,Fp16OutputView) override;
    bool Healthy() const noexcept override;
    std::unique_ptr<LinearWorkspaceScope> BeginWorkspaceReuse() override;
    void Prepare(const Tensor&,size_t rows) override;
    void Forget(const Tensor&) override;
    std::shared_ptr<SpatialConvWeight> PrepareSpatial(const SpatialConvSpec&,const uint16_t*);
    void WarmupSpatial(const std::shared_ptr<SpatialConvWeight>&,size_t height,size_t width);
    void RunSpatial(const std::shared_ptr<SpatialConvWeight>&,const uint16_t*,uint16_t*,size_t batch,size_t height,size_t width,
                    const uint16_t* bias=nullptr);
    Tensor PrepareDynamic(const Tensor&,const std::vector<size_t>& rows) override;
    size_t AttentionTileRows(size_t keys) const override;
    bool SupportsAttentionFp16(size_t keys,size_t head_dim) const override;
    bool RunAttentionFp16(const Tensor&,const Tensor&,Fp16InputView,
                         Fp16OutputView,const AttentionMask&) override;
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
// A logical GEMM uses the shared TFDL Grid policy over eight logical shards.
// Pair-sized tasks migrate between big/small workers using matching commands.
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

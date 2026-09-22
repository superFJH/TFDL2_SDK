#pragma once
#include "tfllm/engine.h"
#include "tfllm/scheduler.h"
namespace tfllm {
struct NpuOptions {
    int chip=0;
    // Per-window split is independently configured by TFLLM_NPU_RUNTIME_MIB.
    double gpu_memory_utilization=0.8; // floor(registered HugePage GiB * fraction).
    size_t max_plans_per_pair=0; // 0: no CPU-template count cap; otherwise chip template limit = value * 4.
    size_t queue_capacity=64;
    int timeout_ms=30000, lock_timeout_ms=30000;
    std::vector<int> worker_cpus; // Four persistent pair workers.
    bool verify_outputs=false;
};
// Also useful for deployment planning without opening an NPU.
uint64_t NpuMemoryBudget(uint64_t hugepage_bytes,double utilization);
struct NpuMemoryStats {uint64_t registered_bytes=0,reserved_bytes=0,used_bytes=0;};
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
    void Prepare(const Tensor&,size_t rows) override;
    Tensor PrepareDynamic(const Tensor&,const std::vector<size_t>& rows) override;
    size_t AttentionTileRows(size_t keys) const override;
    void PrepareModel(const Model&,size_t max_rows=1024);
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
// needs no device lease; execution takes its pair only. Idle engines retain a
// bounded, pressure-evictable instance cache and no driver locks. Every in-flight
// instance has independent mutable commands, A/C and temporary K/V workspace.
std::shared_ptr<Linear> CreateNpuLinear(NpuOptions options={});
std::shared_ptr<Backend> CreateNpuPrefill(NpuOptions options={});
using NpuPoolOptions=NpuOptions;
struct NpuPrefillPool {
    std::shared_ptr<Backend> backend;
    std::shared_ptr<NpuScheduler> scheduler;
};
NpuPrefillPool CreateNpuPrefillPool(NpuPoolOptions options={});
}

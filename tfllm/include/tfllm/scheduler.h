#pragma once
#include "tfllm/engine.h"
#include <functional>
#include <future>

namespace tfllm {
struct LinearWorker {
    std::string name;
    // Called on the persistent worker. Its Linear is used and destroyed on
    // that same thread: SDK core locks cannot move between CPU threads.
    std::function<std::shared_ptr<Linear>()> create;
    int cpu=-1; // Optional Linux CPU affinity; -1 inherits caller affinity.
};
struct WorkerStats {
    std::string name;
    uint64_t completed=0, failed=0;
    // Collected only in _profile builds. Zero means unmeasured in normal builds.
    double queue_ms=0, execution_ms=0;
    bool online=false, busy=false;
};
struct PoolStats {
    uint64_t logical_completed=0; // Whole-chip facade; workers count pair tasks.
    size_t queued=0, peak_queued=0, active=0, peak_active=0;
    std::vector<WorkerStats> workers;
};
class LinearPool final : public Linear {
public:
    // A shared FIFO. The next available worker takes the oldest GEMM.
    // Full queues reject with an exception; accepted jobs drain on shutdown.
    LinearPool(std::vector<LinearWorker> workers,size_t queue_capacity=64);
    ~LinearPool();
    std::vector<float> Run(const Tensor&,const std::vector<float>&,size_t rows) override;
    Activation RunFp16(const Tensor&,const Activation&,size_t rows) override;
    // A group of independent, prebuilt pair tasks. An idle chip can run all
    // tasks concurrently; competing groups share workers fairly at task boundaries.
    // Waits for EVERY task, including after failure, before releasing captures.
    void RunTasks(std::vector<std::function<void(Linear&)>> tasks);
    bool Healthy() const noexcept override;
    PoolStats Stats() const;
    LinearPool(const LinearPool&)=delete;
    LinearPool& operator=(const LinearPool&)=delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Validate whole, disjoint physical pairs; e.g. {0,2,4,6} -> four groups.
std::vector<std::vector<int>> NpuPairDevices(const std::vector<int>& first_cores);

struct SessionExecutorOptions {
    size_t threads=4, max_pending=64;
    std::vector<int> cpus; // Optional, one distinct Linux CPU per worker.
};
class SessionExecutor {
public:
    explicit SessionExecutor(SessionExecutorOptions options={});
    ~SessionExecutor();
    // Different sessions execute concurrently. Requests for the same Session
    // run in submission order; queued requests retain Session/model/KV owners.
    // Do not mix executors or direct calls for one Session if submission order
    // matters. Session's mutex still prevents concurrent KV modification.
    std::future<std::vector<float>> Prefill(std::shared_ptr<Session>,std::vector<int32_t> prompt);
    std::future<std::vector<float>> Append(std::shared_ptr<Session>,std::vector<int32_t> tokens);
    SessionExecutor(const SessionExecutor&)=delete;
    SessionExecutor& operator=(const SessionExecutor&)=delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

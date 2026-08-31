# GELab 外部 KV 连续批处理可行性验证

日期：2026-08-31  
模型：GELab-Zero-4B Q4_K_M GGUF（本机 ARM CPU，32 threads）

## 目标

验证不同时间到达的外部 prefill KV 能否在一个 llama.cpp context 内保留为独立 sequence，并在后续 decode step 合并为一次 `llama_decode()` 的 batch MatMul。

## Probe

新增的 `llmdecode_batch_probe` 不接入 HTTP/API，也不修改正式 `llmdecode` worker：

1. 创建一个 `n_seq_max=B`、`n_batch=B` 的 shared llama.cpp context，其中 `B` 是活跃请求数。
2. 为每条 sequence 构造不同的、有限 FP16 外部 K/V 值；几何均为 GELab 的 `[36 layers, 8 KV heads, S=16, D=128]`。
3. 先将 A 导入 `seq_id=0`，独立 decode 32 steps。
4. 再将其余请求导入 `seq_id=1..B-1`，不清除 A 的 live cache。
5. 每一步从每条活跃 sequence 提交一条 token：`llama_batch.n_tokens=B`，且 batch 内条目具有不同 `seq_id`。这使权重 MatMul 的 M 维为 `B`，而不是 `B` 个独立 `M=1` 进程。

运行命令：

```bash
LLMDECODE_KLEIDIAI_SOURCE_DIR=/tmp/llmdecode-bench-build/_deps/kleidiai-src \
  bash Example/LLMDecode/scripts/build_arm.sh /tmp/llmdecode-bench-build

/tmp/llmdecode-bench-build/bin/llmdecode_batch_probe \
  --model Example/GELab-Zero-4B/deploy/model/decode/llmdecode-q4_k_m/gelab-zero-4b-q4_k_m.gguf \
  --layers 36 --kv-heads 8 --head-dim 128 \
  --prompt-tokens 16 --warmup-steps 32 --shared-steps 64 --sequences 8 --threads 32
```

## 结果

均使用 32 ARM CPU threads、16-token 外部 KV prefix、A 先到达并独立生成 32 tokens，随后其余请求到达；合批阶段每条 request 生成 64 tokens。结果为实际 GELab Q4_K_M 权重上的 decode 测量。

| 活跃请求 B | 合批总吞吐 | 相对单流 | 平均每请求吞吐 | 合批阶段耗时 |
|---:|---:|---:|---:|---:|
| 1 | 16.41 tok/s | 0.99× | 16.41 tok/s | 3.900 s |
| 2 | 28.55 tok/s | 1.71× | 14.28 tok/s | 4.483 s |
| 4 | 52.86 tok/s | 3.20× | 13.21 tok/s | 4.843 s |
| 8 | 85.04 tok/s | 5.11× | 10.63 tok/s | 6.020 s |
| 16 | 119.08 tok/s | 7.18× | 7.44 tok/s | 8.599 s |
| 32 | 146.14 tok/s | 8.97× | 4.57 tok/s | 14.014 s |
| 64 | 163.70 tok/s | 9.88× | 2.56 tok/s | 25.022 s |
| 128 | 174.01 tok/s | 10.46× | 1.36 tok/s | 47.077 s |
| 256 | **176.91 tok/s** | **10.71×** | 0.69 tok/s | 92.614 s |

在此次 `B=1..256` 扫描中，**176.91 tok/s** 是最高总吞吐。曲线在 128 路后几乎饱和：128→256 路只增加 1.7%，但每条请求的生成速率减半。因此它是硬件带宽/计算的吞吐上限测量，不是在线服务应采用的排队深度。

服务建议把 continuous-batching 上限设为 **8**：总吞吐已达 85.0 tok/s，且每请求仍约 10.6 tok/s；若仅追求后台离线吞吐，可设 32（146.1 tok/s）。

llama.cpp 日志确认每条 sequence 的独立导入：先向 sequence 0 导入 A 的 16-token KV，再向其余 `seq_id` 导入不同 KV；各序列在后续合批 decode 中均持续生成，没有互相覆盖 cache。

## 结论

外部 KV 并不阻碍 llama.cpp continuous batching。所需实现是一个**共享 model/context 的调度器**：

- NPU 锁内完成 vision + prefill，并把每个有效 KV prefix 入 decode 队列；
- 分配一个 llama `seq_id`，调用 `llama_memory_import_kv_f16()` 导入该请求的有效 KV；
- 每个 decode tick 从所有活跃 sequence 各取一个 token，组成 `llama_batch(n_tokens=B)`；
- 按 batch index 读取 logits，独立采样、SSE 回调、遇 EOS 后 `llama_memory_seq_rm()` 回收该 `seq_id`。

本 probe 使用合成且彼此不同的 KV，仅验证共享 context、外部 KV 导入和 batch 调度的 ABI/性能路径；不验证回答质量。正式集成前还应使用两个真实 GELab prefill KV 做逐 token 对齐测试。

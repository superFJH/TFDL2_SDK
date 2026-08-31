# LLMDecode：外部 KV 的 ARM CPU decode

这个项目把当前 GELab 的异构推理链拆成明确的边界：

```
TFDL NPU prefill → FP16 K/V [B,Hkv,S,D] + 首 token logits
                                      │
                                      ▼
                     llama.cpp + KleidiAI CPU decode
```

它不再让 CPU 重跑 prompt/prefill；`llama_memory_import_kv_f16()` 直接将
NPU 的每层 FP16 KV 前缀导入 llama.cpp memory。首个生成 token 取 TFDL
prefill 的 `last_token_logits.npy`，随后每个 token 只执行 CPU decode。默认
保持外部 KV 为 llama.cpp FP16 cache，作为精度基线；`--kv-cache-type q8_0`
可启用实验性的量化 KV 路径。

## Python 安装包与动态有效长度

生产接口是 `tfdl_llmdecode.ExternalKvDecodeWorker`。它启动一次
`llmdecode --serve`，在 API 启动阶段常驻加载 GGUF；每次请求只新建一个
llama context 并导入该请求的 KV。构建目标机器 wheel：

```bash
PYTHON=.venv-tfdl-linux/bin/python \
  bash Example/LLMDecode/scripts/build_wheel_arm.sh
python -m pip install --no-deps Example/LLMDecode/dist/tfdl_llmdecode-*.whl
```

`S_valid` 绝不是预设 256。GELab 每次从请求 `attention_mask.sum()` 取得
有效长度；即使 NPU prefill 的物理 bucket 是 `S=256`，导入前也固定执行：

```text
KV storage [B, capacity, Hkv, D]
  -> [:, :S_valid, :, :]
  -> transpose [B, Hkv, S_valid, D]
  -> contiguous NPY -> llama_memory_import_kv_f16
```

因此右侧 pad 区域从不读入 native worker。MRoPE 的三个 position 轴也只由
native 按同一个 `S_valid` 截取；`n_ctx` 由 `S_valid + max_new_tokens + 1`
动态建立。worker 以 JSONL 回传 token ID，Python 可立刻转成 OpenAI SSE
文本 delta。

## 已实现的 ABI

`third_party/llama.cpp/include/llama.h` 增加：

```c
bool llama_memory_import_kv_f16(
    llama_memory_t mem, llama_seq_id seq_id,
    const struct llama_external_kv_f16 * source);
```

输入必须为每层 C-contiguous FP16 `K/V[B=1,Hkv,S,D]`。导入时仅使用一个
`Hkv*D` 的行缓冲完成 `H,S,D → S,H,D` 变换，不创建全量 token-major KV
副本。实现明确要求：

- llama.cpp K/V cache 为 FP16（默认）或实验性的 `Q8_0`；
- FlashAttention 开启（因此 V 是外部 KV 对应的非转置 token-major 存储）；
- GGUF 的层数、KV head 数和 head_dim 与 TFDL prefill 一致；
- prefill 和 GGUF 使用**完全同一份 decoder 权重、RoPE 与 tokenizer**。

这不是 llama.cpp session blob 的 hack，导入后仍可使用正常 `llama_decode`
和标准 KV metadata；因此后续可安全实现多 worker 的连续批 decode。

## ARM / KleidiAI 构建

```bash
cd /path/to/TFDL2_SDK
bash Example/LLMDecode/scripts/build_arm.sh
```

脚本打开 `GGML_CPU_KLEIDIAI=ON` 和 `GGML_NATIVE=ON`。首次配置会由
llama.cpp CMake 下载其锁定的 KleidiAI `v1.24.0` 源码；离线机器请先在
联网构建机完成 build 或提供 CMake 的下载缓存。发布时只需携带
`build/bin/llmdecode` 及其动态库，不需要原始 safetensors。

可在导出机将同一份 HF decoder 转成匹配的 F16 GGUF，或直接转成用于 ARM
decode 的 Q8_0（W8）GGUF：

```bash
PYTHON=.venv-tfdl-linux/bin/python \
  bash Example/LLMDecode/scripts/convert_hf_to_gguf.sh \
  /path/to/GELab-Zero-4B-preview /models/gelab-decoder-f16.gguf

PYTHON=.venv-tfdl-linux/bin/python \
  bash Example/LLMDecode/scripts/convert_hf_to_gguf.sh \
  /path/to/GELab-Zero-4B-preview /models/gelab-decoder-q8_0.gguf q8_0
```

转换前先确认 upstream converter 能识别该 checkpoint 的 `model_type`；若
checkpoint 含视觉 tower，给本项目的 `--model` 必须仍是带 Qwen3-VL decoder
与其 RoPE metadata 的完整 GGUF，不能仅取 language safetensors 拼一个普通
Qwen GGUF。

## 用 GELab 已有 prefill 结果验证

先准备与 GELab decoder 完全相同的 GGUF（不要把量化后的不同权重拿来和
FP16 TFDL KV 混用），然后：

```bash
python Example/LLMDecode/python/run_gelab_external_kv.py \
  --model /models/gelab-decoder-f16.gguf \
  --prefill-dir Example/GELab-Zero-4B/deploy/var/jobs/JOB/prefill \
  --prompt-dir Example/GELab-Zero-4B/deploy/var/jobs/JOB/prompt \
  --kv-cache-type fp16 --max-new-tokens 128 --threads 32
```

该脚本读取现有 `manifest.json`，把 36 对 `.npy` KV 文件 mmap 给 native
程序，并采用 `first_decode_position = valid_seq_len + rope_delta`，与当前
ORT decode 的文本 RoPE 位置规则一致。它还直接读取 prompt 的
`position_ids_3d.npy`；native 仅在内存中补 Qwen3-VL MRoPE 所需的固定第
四轴 0。整个过程不写出第二份 KV。

对于 GELab 的每 token KV 行，`Hkv×D = 8×128 = 1024`，恰好是 Q8_0 的
32 元素 block 的整数倍。Q8 导入器只使用一个 FP16 行、一个 FP32 行和一个
Q8 行临时缓冲；不会形成整份 token-major FP16 KV 副本。`Q8_0` 每 32 个值
额外保存一个 FP16 scale，因此 K/V storage 相较 FP16 约缩小 1.88 倍，且后续
llama.cpp decode 产生的新 K/V 直接追加到同一份 Q8_0 cache。该路径应先做
输出精度对齐，未对齐前不要用于生产。

## 与 API server 的接法

第一版命令行工具用于逐步校验 KV ABI、logits 和 token 对齐。生产接入应在
`api_server.py` 中常驻维护 `llmdecode` worker 池：vision/prefill 继续保留
NPU 互斥锁；完成 prefill 后把 request-owned KV 交给一个空闲 CPU worker。
每 worker 有自己的 llama context/KV memory，从而可让 decode 真正多并发，
不会发生 ORT 目前的全局 generation lock 串行问题。

在导入 KV 后第一步应对比：NPU `last_token_logits` argmax、GGUF prefill
argmax（离线对照）和 llama.cpp 首次 decode logits。三者完全对齐后才应替换
线上 ONNX decode；不对齐通常意味着 GGUF 转换、RoPE metadata 或权重精度
不一致，而不是 KV 字节序。

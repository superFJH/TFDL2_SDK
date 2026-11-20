# Breeze-TTS-2 语音合成

`tfllm speak` 与 `tfllm serve --task tts` 使用 Breeze 专用 C++ runtime，接受
`general.architecture=breeze-tts-2` 的 GGUF，或原始
`BreezeForConditionalGeneration` HF 检查点。运行覆盖 T5Gemma2 文本编码器、
Qwen3 主干、每帧 15 步深度解码、Qwen3TTSTokenizerV2 声码器与参考音频编码。
输出为单声道 24 kHz PCM16 WAV，HTTP 另支持流式 PCM16LE。

## 构建与安装

沿用固定版本 llama.cpp 中的 ggml；模型加载不调用 `llama_model_load_from_file`。
`TFLLM_WITH_BREEZE` 默认跟随 `TFLLM_WITH_LLAMA`，可以单独关闭。

```sh
sh TFLLM/tools/fetch_llama.sh
cmake -S TFLLM -B build/tfllm-tts -DCMAKE_BUILD_TYPE=Release \
  -DTFLLM_WITH_LLAMA=ON -DTFLLM_WITH_BREEZE=ON
cmake --build build/tfllm-tts -j4
```

板端沿用 SDK 的 `WITH_TFLLM=ON` / `TFLLM_WITH_NPU40T=ON` 构建及安装。
普通/profile 两套库均包含 Breeze 入口。SDK 安装会带上转换脚本、固定版本 gguf-py
及第三方源码许可证；更新 pip wheel 时仍需重装完整 SDK。

已有 GGUF 的合成不需要 PyTorch、Transformers、NumPy 或音频 Python 库。
从 HF 导入时安装 `python -m pip install './SDK/tfllm[tts]'`（替换为实际安装目录），
或安装 NumPy、PyYAML、tqdm；下载仓库 ID 另需 huggingface-hub。

## 直接运行 Q8 GGUF

```sh
tfllm speak /models/breeze-tts-2-q8_0.gguf \
  --text '你好，欢迎使用语音合成。' \
  --instruction '自然清晰的年轻女声，语气亲切。' \
  --output hello.wav
```

默认使用 NPU；本机软件验证显式加 `--backend cpu`。CPU 模式保留 GGUF 的原始
权重编码。模型架构、参数量、文件边界、主要 Transformer 张量形状及词表会在加载时检查。

参考音频克隆：

```sh
tfllm speak /models/breeze-tts-2-q8_0.gguf \
  --text '这是需要合成的新句子。' \
  --ref-audio reference.wav --ref-text '这是参考录音的准确文字稿。' \
  --instruction '清晰自然地说话。' --cfg-scale 3 --output clone.wav
```

参考 WAV 最多 30 秒、16 MiB，支持常见 PCM/FP32 WAV，合并声道并线性重采样至
24 kHz。`--cfg-scale 1` 运行单分支；更高值使用条件/无条件双分支。
`--max-tokens` 是每段音频的帧数上限，每帧 80 ms，默认 750 帧。
长文本沿用上游分句及参考码本续接。CLI 完成后才原子发布输出 WAV。

## 从原始 HF 导入和量化

```sh
tfllm speak /models/Breeze-TTS-2 \
  --tts-quantization q8_0 --text '你好。' --output hello.wav

tfllm speak BreezeBlue/Breeze-TTS-2 --model-source huggingface \
  --tts-quantization q8_0 --text '你好。' --output hello.wav
```

HF 目录必须包含 `config.json`、`tokenizer.json`、主模型 safetensors 和
`audio_tokenizer/` 的配置、safetensors。支持 shard index 和 HF snapshot 的 shard
symlink。主干参数取 `backbone_config`；音频编码器与声码器取 `audio_tokenizer/`。
主检查点内的旧 Mimi codec 不用于合成。

导出默认将兼容的二维矩阵直接从 BF16/F32 量化为 ggml Q8_0，不先经过 F16。
Norm、bias、码本、卷积、depthwise 核及三维深度输出头保持 F32；不兼容 Q8 块宽的
矩阵使用 F16。`--tts-quantization f16/f32` 可用于精度对照；
`--depth-quantization f16/f32/q8_0` 只覆盖深度部分的普通矩阵。
`preserve` 表示沿用所选全局导出精度，敏感 F32 张量始终保留。

转换先收集 metadata，再逐张量写入同一临时输出文件，不缓存整个模型，也不另存一份
全模型 F16 中间文件。完成后原子发布到 `--cache-dir/breeze/<key>/model.gguf`，
同目录的 `breeze.json` 记录策略。缓存键包含源文件身份、转换实现及量化选项；
复用缓存不会改写原始权重。直接打开已有 GGUF 时不重新量化文件。

也可独立导出：

```sh
python TFLLM/adapters/breeze/convert.py /models/Breeze-TTS-2 \
  --dtype q8_0 --output breeze-q8.gguf
```

## NPU 执行范围与精度

默认 `--npu-stages te,bb` 接管文本编码器和主干的静态 Linear，包含主干逐帧 decode。
`--npu-stages te,bb,dd` 可进一步接管深度解码矩阵；`codec` 可用于声码器/参考编码器的
Linear 实验。阶段名必须来自 `te,bb,dd,codec`。

接管矩阵会按 GGUF 实际编码解码一次，准备成 TFLLM 逐行对称 UINT8，执行 W8A8，
输入和输出使用 FP32 接口。**这是一次额外重量化，不是原 Q8_0 block 的零复制执行。**
每个模型复用已准备的静态权重；每次图执行复用有界命令和工作区。
模型持有跨请求的深度图及声码器实例池，各请求重置音频/KV 状态。Embedding、Norm、
非线性、Attention 和卷积目前继续由 ggml CPU 执行，F32 深度输出头也保留 CPU 路径。
深度矩阵默认留在原始 ggml 路径，启用 `dd` 后需要特别检查长音频的误差累积。

上游深度解码的 15 步融合图及其 replay、CFG 分支隔离、每帧 KV 重置、声码器卷积历史和
滑窗 KV 都保留。当前一个 engine 串行处理请求；HTTP 并发请求进入有界等待队列，
尚未接入上游跨请求合批调度器。上游 GPU 性能数据不能用作 NPU40T 性能结论。

## HTTP 与统计

```sh
tfllm serve /models/breeze-tts-2-q8_0.gguf --task tts \
  --served-model-name breeze --host 127.0.0.1 --port 8118

curl http://127.0.0.1:8118/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"breeze","input":"你好。","instructions":"温柔自然的女声","response_format":"wav"}' \
  --output hello.wav
```

`response_format=pcm` 使用 HTTP chunked streaming，内容是单声道 24 kHz PCM16LE。
支持 `voice=breeze/default`，音色描述用 `instructions`，采样参数包括 `cfg_scale`、
`temperature`、`top_k`、`top_p`、`seed`、`max_tokens`。`/health` 和 `/v1/models`
可查询状态。沿用 `--api-key`；队列长度取 `--max-pending-requests`。
HTTP 不接受服务器本地参考音频路径，参考音频克隆使用本地 `speak` 命令。

CLI 每次请求输出 `TFLLM_TTS_REQUEST` JSON，包含首音频时间、文本编码、参考编码、
prefill、主干、深度、声码器耗时、帧数、时长及 NPU Linear 调用数。

```sh
tfllm_profile speak /models/breeze-tts-2-q8_0.gguf \
  --text '你好。' --output hello.wav --profile /tmp/breeze-profile
```

profile 版输出 CSV/JSONL；记录请求及接管 Linear/NPU 的算子跨度，阶段总耗时仍在 JSON
统计中。CPU ggml 各节点目前没有独立 trace。普通版不启用 profile。

## 验证边界

```sh
cmake -S TFLLM -B build/tfllm-tts-test -DCMAKE_BUILD_TYPE=Release \
  -DTFLLM_WITH_BREEZE=ON -DTFLLM_BUILD_TESTS=OFF -DTFLLM_BREEZE_BUILD_TESTS=ON
cmake --build build/tfllm-tts-test -j4
ctest --test-dir build/tfllm-tts-test -R '^tfllm-breeze-' --output-on-failure
```

独立 Breeze 测试开关允许在其余可选 fixture 不完整时验证本模块。
HF/端到端测试需 NumPy 与固定版本 gguf-py；不下载真实模型。
测试覆盖 BF16/F32 到 Q8、F32 敏感张量、Gemma Norm 偏移、深度精度覆盖、源快照不变、
缓存复用、四阶段生成、参考音频、CFG、取消及恢复、HTTP WAV/PCM、profile、截断文件、
图 replay 和流式声码器。桥接还与独立 ggml 输出比较，并通过注入 SDK 验证 NPU 调度、
量化、权重复用和错误传播。GPU embedding 对照在没有 GPU 后端时明确跳过。

2026-11-17 另已下载并校验 `xuran4502/Breeze-TTS-2.cpp` 的真实 Q8 文件
（3,568,844,480 字节，SHA-256
`a02bcc4b69b0601032727f8040c4942149b1b73aa0f69022fe5aaa6a8f0ef879`），
在本机 4 线程 CPU 路径完成英文、中文、中文 CFG=3 三条合成，分别输出
2.88、2.00、2.16 秒 WAV。长度、非静音、有限值及格式检查通过。
原始统计与音频见 [本机验证记录](artifacts/breeze-native-20261117/report.json)。

真实文件检查还确认：decoder metadata 的 `codebook_dim=512` 并不是 RVQ 码本的宽度；
码本实际为 256 维，投影到 512 维。加载器按投影张量检查码本形状，回归 fixture 也覆盖了
配置字段与实际 RVQ 宽度不同的情况。

这些测试不包含听感/ASR 可懂度评分、长音频稳定性或 NPU40T 板端性能验收。
真实完整 HF 检查点的导出尚未全量运行；HF 转换已通过小模型四阶段端到端验证。

源码参考：

- [Breeze-TTS-2.cpp，固定提交 155e608](https://github.com/HoppouAI/Breeze-TTS-2.cpp/tree/155e60818433ea6322367aff8f8436ba790f6fbe)
- [原始模型](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
- [官方 PyTorch 推理，核对提交 a43d19e](https://github.com/breezeblue-ai/breeze-tts/tree/a43d19e718b322e1cb3b2aa66f6d37d7945f4b21)

第三方代码为 Apache-2.0；模型权重和自部署输出的授权沿用原模型许可证。

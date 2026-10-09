# Qwen3-ASR 原生接入

`tfllm transcribe` 和 `/v1/audio/transcriptions` 共用原生 Qwen3-ASR 推理链路，支持 dense Qwen3-ASR-1.7B / 0.6B 的受支持配置。音频模型不经过 PyTorch，也不调用整模型 llama ASR 图。

## 推理路径与精度

1. Python 使用 SoundFile 解码文件、多声道取平均，SciPy 重采样到单声道 16 kHz。WAV、FLAC、OGG 可用；MP3 支持取决于安装的 libsndfile。
2. 原生 C++ 生成 Whisper 风格的 128-bin log-Mel。遵循官方 ASR processor 的 `padding=True` / `truncation=False`；本接口逐条编码，按录音实际长度计算，丢弃最后一帧 STFT 后再归一化，特征长度为 `samples // 160`。
3. 原生音频编码器按 100 帧切卷积分块，执行三层 stride=2 Conv2d、精确 erf GELU、sinusoidal position 和窗口双向 Attention。默认 Attention 窗口为 800 个 Mel 帧；有效 token 合并后执行 post norm 与 projector。
4. 音频 embedding 注入语言 decoder 的 `<|audio_pad|>` span。语言 Attention 为 causal，连续 T/H/W 位置相同，不使用视觉 DeepStack。原生 prefill 与 llama CPU decode 共享 KV。

**默认 NPU 链路为：音频编码 W8A8 → TFLLM 语言 prefill W8A8 → llama CPU decode Q4_0（W4），共享 KV 保持 FP16。** NPU Conv2d、Linear 和 Attention 使用现有 W8A8 后端；LayerNorm、GELU、残差及算子边界保持 FP16。语言模型包遵循 TFLLM 原有逐行 UINT8 权重策略。

HF 导入先导出原始浮点音频 mmproj 与 F16 语言 source GGUF，再直接从 source 生成原生 UINT8 包和独立的 `decode-q4_0.gguf`。两份 GGUF 的身份写入模型包，llama 自动选择配对的 Q4_0 文件。带正确 ASR metadata 的 F16/F32/BF16 decoder GGUF 也采用这项默认策略；已量化的 GGUF 按用户选择的原格式保留。启动应显示 `TFLLM decode: weights=q4_0 ...`，无需额外量化参数或安装 `llama-quantize`。

缓存键已更新：从原 HF 权重目录或浮点 GGUF 路径启动会生成新的 W4 配对缓存并在以后复用，旧缓存目录保留。显式传入旧的已准备目录仍按它自己的 manifest 选择权重；切换新默认请传原模型路径。

[W4 decode 实验](artifacts/qwen3-asr-decode-w4-20260928/report.md) 在 15 条录音上未发现新增字符错误，本机 CPU decode 约快 2.1–2.4 倍。这是小样本软件验证；[音频量化工具](ASR_QUANTIZATION.md) 与 prefill 隔离实验仍固定 F16 decode，以便分开测量误差。

Linear 工作区和融合 Attention scope 覆盖整次音频编码，跨窗口、head 和层复用匹配几何，编码结束归还运行区。Conv 执行实例也使用这项 scope：同一权重与几何的分块复用输入、输出、scratch 和私有命令空间，只更新量化参数与 bias，不重新复制命令或重定位地址；不同请求各自拥有实例。scope 结束或缓存压力淘汰时归还工作区与命令空间。静态音频权重随 engine 保留。每个 engine 的音频特征缓存最多 64 条、128 MiB；缓存比较完整波形，身份同时包含音频模型，不同录音不会复用同一音频 KV span。

### Conv bias 与后处理

三层音频 Conv 均为 **3×3、stride=2、padding=1**。默认 NPU 路径把 FP16 bias 按输入与逐通道权重 scale 转成 INT32，在卷积累加器中相加；反量化回 FP16 后执行精确 erf GELU。输入通道拆分时只在第一块加入 bias，后续块的 bias 为零。动态量化参数和 bias 写在私有执行实例中，不修改共享 CPU 模板。

2026-09-29 本机软件对照保持音频 W8A8、原生语言 prefill W8A8 和 llama Q4_0 decode 不变，只移动 Conv bias。15 条录音、371 个参考字符的 CER 为 **3.23% → 2.70%**；14 条转写完全一致，1 条减少 2 字错误，无样本退化，因此默认启用。音频 embedding 不逐位相同；这是单一说话人的小样本验证，尚未测本次优化的板端延迟。逐条结果与误差见 [Conv bias 验证报告](artifacts/qwen3-asr-conv-bias-20260929/report.md)。

CPU 后端或关闭 NPU bias 时，使用多线程 NEON bias＋GELU 融合处理，保留原来的两次 FP16 舍入边界。以下变量在启动服务前设置；模型和量化文件无需重新转换：

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `TFLLM_NPU_CONV_BIAS` | `1` | `0` 使用 Conv 后的 CPU FP16 bias，便于精度/速度 A/B |
| `TFLLM_ASR_CONV_FUSE_POST` | `1` | CPU 后处理时，`0` 分开执行多线程 bias 与 GELU，便于 profile 拆分 |

这两个变量只接受 `0` 或 `1`。NPU bias 开启时，CPU 后处理仅有 GELU；第二个变量不影响这条路径。现有 `TFLLM_NPU_WORKSPACE_REUSE=0` 可关闭实例复用，缓存容量沿用普通 prefill 的工作区预算。

本次新增 SDK 的 `tfaccUpdateConvolutionExecutionQuantization` 接口，板端需同时更新 TFEngine/TFLLM 的普通版与 profile 版 SDK 并重启服务，不能只替换 Python 文件。

## 构建与依赖

沿用 README 的板端构建方式，开启 `TFLLM_WITH_LLAMA=ON`、`TFLLM_WITH_NPU40T=ON` 后安装 SDK。新增代码会自动进入普通版和 profile 版。

```sh
python3 -m pip install -r SDK/tfllm/share/tfllm/python/requirements.txt
# 首次从 HF safetensors 导入时还需要固定版本转换器依赖：
python3 -m pip install -r SDK/tfllm/share/tfllm/llama/requirements/requirements-convert_hf_to_gguf.txt
```

安装树也提供 `python3 -m pip install '.[audio]'`。转换完成后，实际转写只需要 native runtime、Jinja2、NumPy、SoundFile 和 SciPy。
ASR 启动时会检查音频依赖，缺失时给出当前 Python 解释器的安装命令。
旧版服务若已显示 ready、转写却返回 `Install audio dependencies` 的 HTTP 500，
在启动命令的同一 Python 环境安装上述依赖后重启即可，不需要重新转换模型。

若启动时报 `No native TFLLM importer for architectures=['Qwen3ASRForConditionalGeneration']`，
实际运行的 Python 导入器缺少 ASR 注册。当前源码已经支持该 architecture；
检查 `type -a tfllm` 与 SDK 的 `share/tfllm/python/tfllm_cli/models.py` 中是否存在
`Qwen3ASRForConditionalGeneration`。重新构建并完成整个 SDK 的安装，确保 Python
前端、转换脚本和原生库一起更新。通过 pip 安装的命令还需在同一 Python 环境
执行 `python3 -m pip install --force-reinstall --no-deps /path/to/SDK/tfllm`，
详见 [SDK 更新说明](PIP.md)。使用 SDK 的 `bin/tfllm` 完整路径可绕过旧 wheel。

模型输入可为 HF 权重目录、HF `snapshots/<revision>` 目录、`models--Qwen--Qwen3-ASR-1.7B` 缓存根目录、仓库 ID、已准备的 TFLLM 目录，或带正确 ASR metadata 的 decoder GGUF 加音频 mmproj。首次导入写入 `--cache-dir`，不会改动源权重。

## 文件转写

```sh
tfllm transcribe /path/to/Qwen3-ASR-1.7B \
  --audio /path/to/meeting.wav --language zh \
  --context 4096 --max-num-seqs 1 --max-tokens 1024
```

省略 `--language` 自动识别语言。可以使用 `zh` / `en` / `ja` 等 ISO code，或官方语言名 `Chinese` / `English` / `Japanese`。`--system '项目名称、专有名词等上下文'` 对应官方 ASR system context，不是参考答案。默认 greedy，`--temperature` 可覆盖。`--response-format json` 输出 `{"text": "…"}`；`verbose_json` 额外返回 language、duration 和 task。转写结果写 stdout，运行日志写 stderr。

本机验证显式加 `--backend cpu`。该路径用软件执行原生编码器与 prefill，不代表 NPU 速度。

## HTTP 服务

```sh
tfllm serve /path/to/Qwen3-ASR-1.7B \
  --served-model-name qwen3-asr --host 0.0.0.0 --port 8119 \
  --context 4096 --max-num-seqs 1 --max-tokens 1024 \
  --max-audio-seconds 120

curl http://127.0.0.1:8119/v1/audio/transcriptions \
  -F model=qwen3-asr -F file=@/path/to/meeting.wav \
  -F language=zh -F response_format=verbose_json
```

请求格式为 multipart/form-data。支持 `file`、`model`、`language`、`prompt`、`temperature`、`response_format`（json/text/verbose_json）。`prompt` 覆盖服务启动时的 `--system`。沿用现有 API key、请求队列、会话数和取消机制。`/health` 与 `/v1/models` 可查询状态。

文件/HTTP 请求上限 32 MiB，单段音频至少 201 个 16 kHz 样本、默认最多 120 秒，可用 `--max-audio-seconds 1..120` 收紧；特征和生成 token 总数还须落在 `--context` 内。本次实现文件转写，不提供 streaming ASR、词级时间戳、SRT/VTT 或 ForcedAligner。ASR 输入通过音频转写接口提交，交互 `chat` 入口用于文本/视觉模型。

## 验证

`tfllm-asr-import` 生成独立的小型 HF checkpoint，测试 HF 缓存根目录与 shard symlink 导入、原生音频编码器对照、Whisper Mel 对照、分块边界、音频缓存隔离、普通/profile 库、文件 CLI 与 multipart HTTP 接口。测试不依赖 Qwen3-ASR 源码或下载权重，需安装 Torch、Transformers、Safetensors、SentencePiece 和音频依赖。

2026-09-28 本机真实 1.7B 已完成自动语言识别与中文转写；19 个独立数值对照及普通/profile 接口回归通过。最初的软件验收只验证一条真实录音，记录见 [原生 ASR 验证报告](artifacts/qwen3-asr-native-20260928/report.md)；后续板端 HTTP 测试见下文。

随后补充了 15 条录音、75 次转写的语言 prefill 精度对照：原生浮点控制与官方输出 15/15 一致；仅语言 prefill W8A8 新增 3 字错误，音频与 prefill 同时 W8A8 新增 2 字；12 条干净录音均无新增字符错误。这是固定官方 FP16 后续 decode 的本机软件实验，方法与范围见 [语言 prefill 精度报告](artifacts/qwen3-asr-prefill-20260928/report.md)。

默认 W4 导入已补充 HF/浮点 GGUF 自动量化、原生权重独立性、缓存复用/失败清理，以及普通/profile 的 CLI/HTTP 验证。真实 1.7B 新默认生成的 Q4_0 文件与上述 90 次 decode 精度实验所用文件逐字节一致，运行库加载成功；记录见 [默认导入验证](artifacts/qwen3-asr-default-w4-20260928/report.md)。

```sh
ctest --test-dir build/tfllm -R tfllm-asr-import --output-on-failure
tfllm_profile transcribe /path/to/Qwen3-ASR-1.7B \
  --audio /path/to/meeting.wav --language zh --max-tokens 1024 \
  --profile /tmp/asr-profile
```

profile 中新增 `audio/log_mel`、`audio/qwen3_asr`、`audio_layer`、`audio.conv1/2/3` span；请求日志增加 `audio_tokens` 与 `audio_ms`。`audio_ms` 包括 Mel、音频编码及 token 展开，`language_prefill_ms` 单独统计语言 prefill。原有 `prefill_ms` 仍表示请求开始到 prefill 完成的总时间，包含排队和音频编码。拆分板端各阶段耗时和核对内存占用时，需保存对应请求日志及 profile。

Conv 额外拆出 `audio.convN.run` 与 `audio.convN.gelu`；CPU bias 路径为 `audio.convN.bias_gelu`，或关闭融合后的 `.bias` / `.gelu`。NPU 后端记录 `conv_input_range`、`conv_bias_quantize`、`conv_template_lookup/build`、`conv_workspace_acquire/alloc`、`conv_input_quantize`、`conv_command_materialize` / `conv_workspace_reuse` 和 `conv_output_dequantize`，回收沿用 `execution_slot_return`。复用事件的 `cache_hit=1`，命令复制字节数为零；父 span 包含子 span，统计时不要重复相加。

同日对 `10.10.12.65:8119` 的 NPU ASR 服务完成 31 次 HTTP 转写：15 条录音合计 80.62 秒，首轮客户端端到端耗时 30.33 秒（RTF 0.376），立即重复为 23.55 秒（RTF 0.292）；14.87 秒长音频耗时 4.92 秒。CER 为 14/371 = 3.77%，比本机联合 W8A8 + W4 对照多 2 字，均来自 10 dB 噪声样本；12 条干净录音仍为 7/253 = 2.77%。本轮只有单一角色说话人，参考来自未人工审核的文件名；服务 HTTP 响应未提供内部计时和 decode 权重信息。逐条结果、限制与复现工具见 [板端 HTTP 报告](artifacts/qwen3-asr-http-20260928-ready/report.md) 和 [asr_http_benchmark.py](tools/asr_http_benchmark.py)。

2026-09-29 补充 1/2/4/8/10 路客户端并发与长录音突发测试，共 265 次请求全部成功。
当前服务 `max_sequences=4`、共享 CPU decode 线程数为 4；新波形混合负载峰值为 3.71 秒音频/秒。
3～6 秒短录音的端到端 P95：单路 2.32 秒、两路 3.59 秒、四路 6.80 秒、十路 16.30 秒。
10 段约 15 秒录音同时提交，整批完成需 38.35 秒。常态一两路、偶发十路允许排队可评估替换专用 ASR 显卡；
持续十路实时语音当前配置不足。更多执行槽位尚未测试，不能据此判定硬件上限。
数据与限制见 [延迟并发评估](artifacts/qwen3-asr-load-20260929/report.md)，可复用压测器见 [asr_http_load_benchmark.py](tools/asr_http_load_benchmark.py)。

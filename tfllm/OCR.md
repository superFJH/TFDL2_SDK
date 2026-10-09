# MinerU / PaddleOCR 接入 TFLLM

TFLLM 可作为两个项目的 OpenAI 兼容 VLM 推理服务。图片编码、projector 和图文 prefill 使用 TFLLM 原生路径，共享 FP16 KV 后由 llama.cpp CPU decode 接力。PDF 渲染、版面检测、裁剪、结果拼接继续由 MinerU / PaddleOCR 客户端完成。

## 对应版本

截至 2026-09-28 核对的官方版本：

| 客户端 / 模型 | TFLLM 结构 | 本次支持 |
| --- | --- | --- |
| MinerU 4.0，`opendatalab/MinerU2.5-Pro-2605-1.2B` | `qwen2vl` + `qwen2vl_merger` | 官方 HF snapshot 导入、图文推理、顶层 32768 上下文覆盖嵌套的旧 8192 默认值 |
| `PaddlePaddle/PaddleOCR-VL-1.6` | `paddleocr` + `paddleocr` projector | 官方原始权重格式及 Transformers 原生嵌套格式、动态图片分辨率、三轴 MRoPE |

模型版本来自 [MinerU 官方注册表](https://github.com/opendatalab/MinerU/blob/master/mineru/model/registry.py) 和 [PaddleOCR-VL-1.6 官方仓库](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)。完整权重的 safetensors 头信息已核对，Paddle 的 Q 投影宽度 2048 与语言 hidden_size 1024 分开处理；视觉 head_dim 为 72。原始 Paddle 模型采用无 antialias 的 bilinear 位置插值、`align_corners=False`，原生 HF 格式采用 `True`，转换器分别保留。

此接入覆盖文档 VLM。MinerU 的 ONNX / Torch 小模型和 Paddle 的版面检测等模块仍需原客户端环境；PP-OCRv6 的传统检测/识别网络、PP-Structure 等独立管线不因此变成 TFLLM 模型。

## 启动 TFLLM

按 [README](README.md#从下载的模型直接聊天或启动服务) 构建、安装并准备 Python 转换依赖。支持完整本地 HF snapshot，也支持直接提供仓库 ID；无需执行模型仓库中的自定义 Python。

MinerU 服务：

```sh
tfllm serve opendatalab/MinerU2.5-Pro-2605-1.2B \
  --revision bff20d4ae2bf202df9f45284b4d43681555a97ed \
  --served-model-name mineru \
  --context 32768 --max-image-tokens 4096 \
  --max-num-seqs 1 --max-tokens 8192 --port 8000
```

PaddleOCR 服务：

```sh
tfllm serve PaddlePaddle/PaddleOCR-VL-1.6 \
  --revision c5630abae1d940eafe0697512a0325494b02ab42 \
  --served-model-name paddleocr-vl \
  --context 32768 --max-image-tokens 4096 \
  --max-num-seqs 1 --max-tokens 8192 --port 8118
```

默认后端为 NPU，启动应显示 `backend=npu40t-chip-fp16` 和 `modalities=text,image`。在主机上做功能检查时显式加 `--backend cpu`。使用本地模型时，把仓库 ID 换成完整 snapshot 目录即可。

## Vision 优化与 score budget 调优

原生视觉 FFN 的 QuickGELU、SiLU、GELU-tanh 和 projector GELU-erf 默认采用 FP16 查表；每种表按当前平台的原算式生成，保留激活的 FP16 舍入。共五张 128 KiB 表（包括 exact GELU 的独立算式），CPU 初始化时生成，进程内只读共享。decoder 的 `SiLU/GELU * up` 仍在乘法后才窄化，不引入提前舍入。split Q/K/V 每 1024 行共享一次输入量化；dense FFN 按块处理，并将激活与 down 输入的完整行量化放在同一 CPU 工作组中。偏置仍是独立的 FP16 边界。

每个 attention scope 可复用最多两套空闲的私有工作区和命令；Dense 和 Qwen3 vision 均在层循环外创建 scope，覆盖整次编码，跨 head、跨层复用匹配几何。Qwen3 的每张静态图片或每个视频帧对分别使用独立 scope。每个在途 head 独占一套，复用时重新写入 K/V、列尺度和 query 输入。命令连续空间不足时先在固定预算和 256 MiB 上限内增长，增长失败或运行区不足时再回收空闲实例。scope 结束归还全部工作区和命令租用，底层命令池容量保留供后续请求使用；pair lease 仍按原 tile 边界释放。

普通 Q/K/V、attention output、FFN 和 projector 的 GEMM 也在图片编码内复用工作区。语言 prefill 的作用域覆盖一次请求的所有分块；同形状、同 HIGH 的静态层更新权重绑定即可沿用 A/C、scratch 和命令。动态 GEMM 每次刷新 K/V 快照和尺度；请求/编码结束归还所有缓存。详情见 [NPU_RUNTIME_SLOTS.md](NPU_RUNTIME_SLOTS.md)。

| 环境变量 | 默认 | 可选值 | 用途 |
| --- | --- | --- | --- |
| `TFLLM_NPU_ATTENTION_SCORE_MIB` | `4` | 整数 `1..16` | 每个 slot、所有 pair lane 合计的 QK INT32 score 预算，单位 MiB |
| `TFLLM_NPU_ATTENTION_FUSE_WORKERS` | `16` | 整数 `1..32` | 每 lane 的 CPU softmax/quantization 参与线程数，含调用线程 |
| `TFLLM_CPU_ACTIVATION_LUT` | `1` | `0` / `1` | 关闭 / 开启 unary 激活查表，用于 A/B |
| `TFLLM_NPU_ATTENTION_WORKSPACE_REUSE` | `1` | `0` / `1` | 关闭 / 开启融合 QK/AV 的工作区复用，用于 A/B |
| `TFLLM_NPU_WORKSPACE_REUSE` | `1` | `0` / `1` | 关闭 / 开启普通 GEMM 的请求/编码作用域复用；语言融合 QK/AV 同时使用该请求作用域 |
| `TFLLM_NPU_WORKSPACE_CACHE_MIB` | `1024` | 整数 `1..4096` | 每作用域空闲普通 GEMM 的工作区＋命令字节上限；最多 32 个实例，按需保留 |

NPU 环境变量在创建 scheduler 时读取，有值时覆盖对应 SDK `NpuOptions`；未设置时保留显式 SDK 参数。激活开关在 CPU 初始化时读取一次。修改后重启服务。空值、非整数和越界值会明确报错；启动 `NPU_ATTENTION` 打印 `score_bytes_per_slot`、`score_MiB_per_slot`、`score_source`、`fuse_workers`、`activation_kernel` 和 `workspace_reuse`。

普通 GEMM 的缓存状态在 `NPU_WORKSPACE` 中打印；profile 可比较 `execution_reserve`、`command_materialize`、`command_pool_grow`、`execution_workspace_reuse`、`execution_weight_rebind` 和 `execution_workspace_evict`。用 `TFLLM_NPU_WORKSPACE_REUSE=0/1` 保持其余参数一致做 A/B；预算可扫描 `256/512/1024` MiB。它不是预分配量，也不会扩大固定 NPU 池；实际容量不足仍会压力回收。命令池实际容量以 `NPU_COMMAND_POOL cached_bytes=...` 为准。

先保持其他参数一致，分别重启测试 `4`、`8`、`16` MiB。例如：

```sh
TFLLM_NPU_ATTENTION_SCORE_MIB=8 tfllm_profile serve PaddlePaddle/PaddleOCR-VL-1.6 \
  --revision c5630abae1d940eafe0697512a0325494b02ab42 \
  --served-model-name paddleocr-vl --host 0.0.0.0 --port 8118 \
  --context 32768 --max-image-tokens 4096 --max-num-seqs 1 --max-tokens 8192 \
  --profile /tmp/ocr-score-8 --profile-max-events 2000000
```

对 5040 个 patch（keys padding 到 5056）、四个 lane 的大页，4 / 8 / 16 MiB 对应每 lane 的 48 / 96 / 192 query rows。两 head、每 head 两 slot 的实际 QK score 合计约 14.81 / 29.63 / 59.25 MiB，另有 K/V、AV、命令和内部 scratch。最少 16 行的几何对齐可能超过很小的预算；它不限制整套工作区，也不改变图片 token 数、可见 key 或识别算法。它与控制每窗口运行区分配的 `TFLLM_NPU_RUNTIME_MIB` 独立。

每轮使用相同图片、像素限制、prompt、输出 token 上限和并发数。优先比较新图片请求的 `VisionEncoder`、`session.prefill`、TTFT、CER/表格结构；相同图片重复请求会命中视觉/前缀缓存，应另行记录。更大的预算可能因 CPU cache、运行区压力或公平调度变慢，最佳值需在目标板实测。确定 score budget 后再扫描 `TFLLM_NPU_ATTENTION_FUSE_WORKERS=4/8/16`，每组启动新进程。

profile 中 MinerU 激活现在标记为 `quick_gelu`；Paddle projector 单列 `mm.gelu_erf`。dense FFN 的激活 span 包含 `activation_quantize`，比较旧版本时同时看整个 `dense_ffn`，不能直接把融合前后的激活 span 当作相同工作量。

安装目录的 CPU 微基准可单独对比激活开关（固定输入、3 次 warmup、9 次中位数，重置输入不计时）：

```sh
TFLLM_CPU_ACTIVATION_LUT=0 /path/to/tfllm/test/tfllm-cpu-benchmark 9 --ocr
TFLLM_CPU_ACTIVATION_LUT=1 /path/to/tfllm/test/tfllm-cpu-benchmark 9 --ocr
```

该微基准覆盖实际 OCR 的 5120 / 4304 / 4608 宽度，并验证融合激活量化的 UINT8 字节和行尺度一致。主机结果只用于内核对照，端到端收益以重编并部署后的板端 profile 为准。

`--context` 必须是 64 的倍数；图片展开后的 prompt 与最大输出长度之和须小于该值。初始并发设为 1，确认板端内存与吞吐后再同时增加客户端并发和 `--max-num-seqs`。BF16 模型导入沿用 TFLLM 的 NPU UINT8 / CPU Q4_0 decode 策略，真实 OCR 精度需在目标设备上复测。

## MinerU 4.0 客户端

安装 MinerU 的 Standard 管线及 ONNX 小模型依赖，在 `mineru-tfllm.yaml` 中设置：

```yaml
model:
  small_backend: onnx
  vlm:
    server_url: http://127.0.0.1:8000/
    model: mineru
    max_concurrency: 1
    http_timeout: 600
```

```sh
export MINERU_CONFIG="$PWD/mineru-tfllm.yaml"
mineru-kit parse input.pdf -o output.md --tier standard
```

指定远端 `model.vlm.server_url` 后，客户端使用 TFLLM 运行 VLM；小模型仍由所选后端加载。地址与配置见 [MinerU 4.0 官方说明](https://github.com/opendatalab/MinerU/blob/master/docs/en/usage/model_source.md)。若启用 TFLLM 的 `--api-key`，在 YAML 的 `model.vlm.api_key` 设置相同值。旧版 MinerU 2.x 的 `vlm-http-client` CLI 与 4.0 的 `mineru-kit` 配置方式不同。

## PaddleOCR-VL 客户端

使用支持 v1.6 的 PaddleOCR 客户端及其本地版面模型环境：

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    device="cpu",  # 本地版面/预处理模块；VLM 在 TFLLM 服务端运行
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://127.0.0.1:8118/v1",
    vl_rec_api_model_name="paddleocr-vl",
)
for result in pipeline.predict("input.pdf"):
    result.save_to_json("output")
    result.save_to_markdown("output")
```

`vllm-server` 选择客户端的 OpenAI HTTP 协议，服务端实际由 TFLLM 执行。客户端用法见 [PaddleOCR 官方文档](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html)。启用鉴权时额外设置 `vl_rec_api_key`。

也可直接检查图片识别：

```sh
tfllm chat /models/PaddleOCR-VL-1.6 \
  --image document.png --context 32768 --max-image-tokens 4096
# 输入 OCR:、Table Recognition: 或 Formula Recognition: 等任务提示。
```

## HTTP 兼容性与分辨率

服务提供 `/v1/models`、`/v1/chat/completions`，支持 JSON 和 SSE。图片使用 `image_url` 的 base64 `data:image/...` 输入，可与文本穿插并提交多图。

支持两项目当前使用的参数：

- `max_tokens` / `max_completion_tokens`：可同时提交相同值，冲突值报错。
- `skip_special_tokens=False`：保留 OCR 结构与坐标等特殊 token，结束 token 仍结束生成。
- `vllm_xargs.no_repeat_ngram_size` 或 `no_repeat_ngram_size`：只约束生成的 token，不包含 prompt；范围 0..16384，0 关闭。MinerU 默认的 100 可直接使用。
- `priority`：int32，较小值优先获得空闲会话槽位，同优先级按到达顺序；不抢占运行中的请求。
- `vllm_xargs.debug`：接受布尔值；TFLLM 的请求诊断使用现有 `TFLLM_REQUEST` 日志。
- `mm_processor_kwargs.min_pixels/max_pixels`：支持 PaddleOCR / Qwen 动态图片预处理，不能超过服务的 `--max-image-tokens` 预算。

PaddleOCR 默认像素范围从官方 preprocessor 导入：112896..1003520，图片按 28 像素的完整 merge 单元缩放，使用 RGB、bicubic resize 和 mean/std=0.5 归一化。服务的图片 token 上限仍是硬限制；最多 4096 个合并 token/图，16 个图片 content parts/请求。显式的最小像素要求超出服务预算时拒绝请求。修改分辨率会影响细小文字识别率。

## 验证与板端验收

```sh
ctest --test-dir build/tfllm-llama \
  -R 'tfllm-(ocr-import|ocr-sampling|chat-frontend)' --output-on-failure
```

`ocr-import` 需 Transformers >=5.17、Torch、sentencepiece、safetensors、Pillow、NumPy 和 Jinja2。测试现场创建三组小权重模型，经真实 HF → GGUF → TFLLM 导入后，与 Transformers 对照图片特征、图文 prefill、共享 KV decode、多图 logits；还检查 UINT8 包生成、前缀缓存、长上下文配置和真实 HTTP 图片/SSE 请求。`ocr-sampling` 使用独立暴力算法验证 n-gram 禁重；`chat-frontend` 验证参数拒绝、图片像素限制、优先级排队和鉴权。

这些是主机 CPU 数值及接口回归，包含受硬件对齐限制的 MatMul 验证；尚未运行两套完整官方权重的 NPU40T OCR 精度、整份 PDF 管线及性能验收。上板后应使用相同文档对照官方后端，分别检查文字、表格、公式、坐标 token 与长页输出，并记录 `TFLLM_REQUEST` 的实际后端和 native token 数。

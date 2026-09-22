# TFLLM

独立于 TFDL 图执行器的 LLM 运行时，接收 GGUF、生成 TFLLM 模型包，按真实长度分块 prefill，并与 llama.cpp 接力 decode。面向本地 0.1B–30B dense 模型；视觉支持 MoonViT、CLIP/SigLIP、InternVL、Pixtral、Qwen2/2.5/3-VL 的指定结构。patch 卷积用 Unfold + MatMul，复用整芯片 NPU 调度，不引入通用 Conv/Pool 图执行器。

## 模型支持范围

按 **GGUF architecture 和实际张量/超参数** 选择实现，不仅匹配文件名。导入时统计所有 GGUF 分片的张量元素，总参数上限为 **30,000,000,000**，MoE 计入全部专家，而非每 token 激活参数量；超过上限在加载权重前拒绝。模型名中标称的“30B”若实际超过此数也会拒绝。

| GGUF 结构 | 代表模型族（限上述参数范围） | 当前执行路径 |
|---|---|---|
| `llama` | Llama 2/3；以 llama 结构保存的 Mistral、Llama 蒸馏模型 | 原生 TFLLM prefill + 共享 KV llama decode |
| `qwen2` | Qwen2、Qwen2.5、对应 Qwen 蒸馏模型 | 原生，支持 Q/K/V bias |
| `qwen3` | Qwen3 dense | 原生，支持 Q/K norm |
| `qwen2vl` | Qwen2-VL、Qwen2.5-VL dense | 原生图文 prefill、连续分段三轴 MRoPE、共享 KV；配套 Qwen2/2.5 mmproj |
| `qwen3vl` | Qwen3-VL dense，如 Qwen3-VL-4B-Instruct | 原生图文/视频 prefill、三轴 MRoPE、DeepStack、共享 KV；配套 `qwen3vl_merger` mmproj |
| `gemma` | Gemma 1 | 原生，embedding 缩放、GELU、共享 embedding/LM head 权重 |
| `gemma2` | Gemma 2 | 原生，额外 Norm、滑窗、attention/logits softcap |
| `gemma3` | Gemma 3 文本 decoder | 原生，Q/K norm、额外 Norm、每层滑窗及 RoPE 参数 |
| 其余官方版本支持的普通自回归文本结构 | 如 Phi3、MoE、混合结构 | 整模型 llama.cpp CPU 回退；不标记为 NPU 已编译 |

原生路径支持标准/linear RoPE、normal/NEOX 排列和 Llama 的频率因子张量。YaRN/LongRoPE、非统一层维度、原生实现未消费的额外张量等会明确回退。回退仍取决于固定版本 llama.cpp 的 CPU 模型支持和内存要求；不保证所有 GGUF、视觉输入、encoder-decoder 或依赖另一个模型的 draft/assistant 文件均可执行。

Qwen3-VL 纯文本的 T/H/W position 相同，三轴 interleaved MRoPE 等价于完整 head 的 NeoX RoPE；文本输入的 deepstack 附加 embedding 为零。导入保留 `qwen3vl` 架构身份、Q/K norm、独立的 head_dim 与共享 embedding/head 权重，prefill 使用原生 TFLLM，短 prefill/decode 使用对应 llama 图。仅接受合法的三轴全 head MRoPE；Qwen3-VL MoE 尚未加入原生路径。加载配套 mmproj 后，图像特征通过 `PromptInput` 注入 embedding/DeepStack；图像 chunk 强制走 TFLLM，后续文本短 prefill/decode 接力 llama。KV 物理槽位与 RoPE T/H/W 分离，追加/回滚按实际 token 槽位管理。

这里的“原生”是已实现结构语义和转换，不代表表中每个真实 checkpoint 都已上板验收。

本轮不扩展 MoE。`serve`/`chat` 的 HF 导入拒绝 MoE，模型包要求 `--require-native`，不会因不支持的视觉或语言结构静默改用整模型 CPU。底层显式 llama fallback 接口保留原有行为。

**当前是可构建、可验证的第一版，不是已完成板端性能验收的 Qwen3-8B 产品。** 本地已验证真实 llama.cpp 共享 KV 接力。NPU40T 路径已实现并做编译检查，尚需在板上运行数值回归和模型精度/性能测试。

## 构建与本地运行

核心库只依赖 C++17、POSIX mmap、线程及本仓库 FlatBuffers 23.5.26 头文件，不链接 TFDL/ONNX/OpenCV。已有生成头文件，正常构建不需要 flatc。

```sh
cmake -S TFLLM -B build/tfllm -DCMAKE_BUILD_TYPE=Release
cmake --build build/tfllm -j4
ctest --test-dir build/tfllm --output-on-failure
./build/tfllm/tfllm-demo
```

接入官方 llama.cpp：源码放在 `Third_Party/llama.cpp`，固定提交见 [REVISION](adapters/llama/REVISION)。源码目录不直接提交到父仓库，由固定版本下载脚本复现。下载脚本不会覆盖已有源码修改；版本不匹配会报错，不能把不同提交当成兼容版本继续构建。

```sh
sh TFLLM/tools/fetch_llama.sh
cmake -S TFLLM -B build/tfllm-llama -DCMAKE_BUILD_TYPE=Release -DTFLLM_WITH_LLAMA=ON
cmake --build build/tfllm-llama -j4
ctest --test-dir build/tfllm-llama --output-on-failure
```

适配器给 `llama_kv_cache` 添加一个明确的 friend 声明，并修正固定版 CPU IMRoPE 的 Qwen3-VL 尾部坐标选择：`[24,20,20]` 的 H/W 交错区域之外继续使用 T，不能选第四个坐标。两处补丁由 [apply_bridge.py](adapters/llama/apply_bridge.py) 校验提交及源码锚点后幂等应用；构建时自动生效，无需手改 submodule。当前 TFLLM 编译的 llama 后端为 CPU，该修正不覆盖外部 llama GPU 后端。不依赖猜测 private 对象布局的强制转换。内部 API 不是 upstream 的稳定 C API，因此必须固定提交。

## 随父项目安装

在父项目中通过 `add_subdirectory(TFLLM)` 接入后，直接执行父项目的 `make install`。所有 TFLLM 文件放在父项目 `CMAKE_INSTALL_PREFIX` 下的 `tfllm/`，不修改父项目的安装前缀：

```text
<install>/
├── lib/                    # 父项目的 libNPU40T.so 等 SDK 库
└── tfllm/
    ├── bin/                # inspect、demo；启用 llama 时还有 convert、run
    ├── lib/                # TFLLM 库和启用的 llama/ggml 静态依赖库
    ├── include/            # tfllm/*.h 和启用的 llama/ggml 公共头文件
    ├── test/               # 已构建的测试、CPU benchmark、NPU/llama 对照程序
    ├── schema/             # model.fbs
    ├── llama/              # 固定版本 REVISION 和许可证（启用 llama 时）
    └── README.md
```

本仓库顶层已加入 `WITH_TFLLM` 开关，在 NPU40T SDK target 创建后接入 TFLLM。保留原来的板端工具链和依赖配置，再添加：

```sh
cmake -S . -B build -DWITH_TFLLM=ON -DTFLLM_WITH_LLAMA=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build -j4
cmake --build build --target install
# Unix Makefiles 下等价于在 build 目录执行 make install
./install/tfllm/bin/tfllm-demo
```

启用 llama 前仍需执行 `sh TFLLM/tools/fetch_llama.sh`。父项目已有 `NPU40T` target 时，TFLLM 默认启用 NPU adapter 并直接链接该 target，无需事先编出 `.so` 再填写库路径。若自行接入，将 `add_subdirectory(TFLLM)` 放在 SDK target 创建之后即可。独立构建仍支持 `TFLLM_NPU40T_LIBRARY` 指定外部 SDK。

`TFLLM_BUILD_TESTS=OFF` 可关闭 host 测试和 CPU benchmark；NPU adapter 启用时仍构建、安装板端测试。启用 NPU 和 llama 时，`tfllm-run-concurrent` 安装到 `bin/`，`tfllm-npu-llama-benchmark` 安装到 `test/`。安装后的 bin/test 通过相对 RPATH 查找 `tfllm/lib` 和父项目的 `lib`，也支持 `cmake --install build --prefix /新的安装目录` 及 `DESTDIR` 暂存。单独构建 TFLLM 时采用相同的 `tfllm/` 子目录布局。

一次 CMake 构建会同时生成普通版与 `_profile` 版；两者分别编译、链接对应的 TFLLM 核心库和 NPU/llama adapter，llama.cpp 本身的静态依赖共用。`make install` 会将两套文件并排安装：

| 用途 | 普通版 | 支持采集的版本 |
| --- | --- | --- |
| 聊天/服务 | `bin/tfllm` | `bin/tfllm_profile` |
| token ID 示例 | `bin/tfllm-run`、`bin/tfllm-run-concurrent` | 同名加 `_profile` |
| 板端模型对照 | `test/tfllm-npu-llama-benchmark` | `test/tfllm-npu-llama-benchmark_profile` |
| 库 | `libtfllm`、`libtfllm-npu40t`、`libtfllm-llama`、`libtfllm-chat` | 各库名加 `_profile`，再加 `.a/.so` 后缀 |

普通版以 `TFLLM_ENABLE_PROFILE=0` 编译，预处理阶段删除 `TFLLM_PROFILE(...)` 内的整个表达式，包括名称拼接、事件元数据、计时对象和跨线程上下文；不链接 `profile.cpp`。普通版拒绝 `--profile`，调度器的 `queue_ms/execution_ms` 为零（未测量），任务数量、公平调度、NPU 超时和锁保护保留。请求级总耗时、benchmark 端到端测速以及 SDK 自己的可选 `TFDL_NPU40T_CONTROL_TRACE_US` 是独立机制。

`_profile` 版以 `TFLLM_ENABLE_PROFILE=1` 编译，逐算子采集仍默认关闭，必须显式传 `--profile`。该版本还保留调度器队列/执行时长统计。正式测速使用普通版。CMake target 会传递正确的宏；外部手工链接时必须使用匹配的库和宏，不能在同一个程序中混用两套 TFLLM 库。

## 从下载的模型直接聊天或启动服务

父项目启用 `TFLLM_WITH_LLAMA=ON`、`TFLLM_WITH_NPU40T=ON` 后重新构建并 `make install`。
新入口安装到 `<install>/tfllm/bin/tfllm`，C++ 推理库、Python 前端及固定版本的 HF 转换脚本随 SDK 安装。
Python 要求 3.9 以上，建议使用虚拟环境；纯文本运行需要 Jinja2，图片解码/预处理另需 Pillow、NumPy（已列入 requirements）；HF 目录首次转换还需要转换器的 Python 依赖。

```sh
SDK=/workspaces/NPU40T/TFDL2_SDK
python3 -m venv /workspaces/tfllm-venv
. /workspaces/tfllm-venv/bin/activate
python -m pip install -r "$SDK/tfllm/share/tfllm/python/requirements.txt"
# 从 config.json + safetensors 首次导入时安装；已有 GGUF 可以跳过。
python -m pip install -r "$SDK/tfllm/share/tfllm/llama/requirements/requirements-convert_hf_to_gguf.txt"
# 仅在线下载需要对应 provider；本地目录无需安装。
python -m pip install modelscope huggingface_hub
export PATH="$SDK/tfllm/bin:$PATH"

tfllm serve /models/Qwen3-8B --gpu-memory-utilization 0.8
tfllm chat /models/Qwen3-8B
tfllm serve Qwen/Qwen3-8B --model-source modelscope
# Hugging Face；未指定 model-source 的仓库 ID 默认使用 Hugging Face。
tfllm serve Qwen/Qwen3-8B --model-source huggingface
# 完整 Qwen3-VL HF snapshot 自动导入 decoder 和 mmproj，默认使用 NPU。
tfllm serve /models/Qwen3-VL-4B-Instruct --gpu-memory-utilization 0.8
tfllm chat Qwen/Qwen3-VL-4B-Instruct --model-source modelscope --image /images/example.jpg
```

本地输入接受完整 Transformers 浮点 safetensors 目录、GGUF 文件（分片传第一片）、只含一个 GGUF 模型的目录，以及已生成的 TFLLM 缓存目录。
`--gguf-file 'model-Q4_K_M*.gguf'` 可选择仓库内指定量化版本并下载其全部分片；多个 GGUF 版本不会凭文件排序任意挑选。
`--revision` 固定仓库版本。下载复用 provider 自己的缓存；本地目录不访问下载 provider。

首次自动执行 HF → 浮点 GGUF → 原生 UINT8 包；已有 GGUF 直接进入第二步。转换使用 `--require-native`，不支持的架构/配置直接报错，禁止生成整模型 CPU 回退包。
对于未量化的 BF16 源模型，首次导入还通过内置 llama 量化 API 生成 CPU decode 专用 `decode-q4_0.gguf`，`serve/chat` 自动加载它，无需增加启动参数或单独安装 `llama-quantize`。NPU UINT8 权重仍直接从 BF16 源 GGUF 生成，不从 Q4_0 再次量化；两份文件的内容身份写入模型包，运行时只接受原始 GGUF 或对应的 decode GGUF，不能混用另一模型。默认选择 Q4_0（32 个权重一组），使用 llama 的量化与 CPU 重排实现，norm/bias 和不满足分块条件的张量采用 upstream 的浮点保留/回退，不能理解为所有张量都只有 4 bit。共享 KV 仍为 FP16。
F16/F32 以及已经量化的 GGUF 保持原有 decode 格式（包括混合低比特/BF16 的 GGUF），不会自动重新量化；本策略也适用于直接输入未量化 BF16 GGUF 及其分片。mmproj 独立导入视觉 NPU 路径，不生成 CPU decode 副本。启动打印 `TFLLM decode: weights=q4_0 threads=... gguf=...`，用于确认选中的 decode 文件。Q4_0 会带来额外量化误差，实际模型需复测精度与速度。
下表支持的 HF 多模态结构可直接传完整 snapshot 路径（包括 `.cache/huggingface/hub/.../snapshots/...`）。默认 `--mmproj auto`：有 `preprocessor_config.json` 或 `processor_config.json` 时，固定版转换器分别提取语言 GGUF 和视觉 mmproj GGUF，后者随缓存一起保存；原版 `InternVLChatModel` 没有 processor JSON 时采用其 ImageNet 归一化默认值。不修改下载目录，也不执行下载模型附带的 Python 代码。已下载的语言 GGUF 会自动发现同目录唯一的 `mmproj*.gguf`，有多个时必须指定 `--mmproj FILE`。`--mmproj none` 仅启用文本。图像模式启动显示 `modalities=text,image`；Qwen3-VL 配套视觉模型另显示 `video`。
浮点 safetensors 导入在读取权重前检查分片、dtype、参数量（不超过 30B）；AWQ/GPTQ/FP8 等量化 safetensors 暂不接受，可以使用当前支持的量化 GGUF。
GGUF 的 Q4/Q8/K-quant 先解码，再重新量化成 NPU UINT8 权重，因此存在额外量化误差及磁盘/内存开销，不等同于 NPU 直接执行 Q4。

转换缓存默认 `~/.cache/tfllm`，可用 `--cache-dir` 或 `TFLLM_CACHE_DIR` 修改。缓存区与来源目录分离，使用进程锁、源文件身份/变更信息、转换器版本和临时目录原子发布；重复启动不重新转换。
HF 缓存目录包含 `source.gguf`、`model.tfllm`、`model.weights`、`model.commands`、`manifest.json`，图像模式还有 `mmproj.gguf`，可以把整个目录复制到板端直接启动，无需保留原 HF 目录。GGUF 输入若保留原 decode 格式，缓存引用原 GGUF，必须保留源文件全部分片。
BF16 导入的缓存另外包含 `decode-q4_0.gguf`，manifest 的 `gguf` 指向它、`source_gguf` 保留原始来源；运行配对 Q4_0 时不需要再加载 BF16 源文件，但保留它可做原始精度对照。新的缓存版本为 3，继续使用原模型路径启动会自动生成 Q4_0 新缓存，不命中旧 Q8_0 导入缓存，旧目录不覆盖；若直接传入版本 1/2 的准备目录，会提示从原模型路径重新导入。首次生成 Q4_0 增加转换耗时和磁盘占用，后续启动复用，量化失败或文件缺失会报错，不会静默回到 BF16。
权重缓存不是完整离线硬件编译产物；NPU 仍按已有 JIT 机制创建、复用命令计划。

默认后端是 NPU。启动应显示 `TFLLM_READY ... backend=npu40t-chip-fp16`；SDK 没有编译 NPU adapter、设备初始化或内存预留失败都会报错，不会自动切成 CPU。
本地功能测试可以显式使用 `--backend cpu`，其日志会显示 CPU prefill 后端。长 prefill 使用 NPU MatMul 和 CPU 算子，默认不超过 32 token 的短块及 decode 仍按既定策略走 llama CPU。

`serve` 默认监听 `127.0.0.1:8000`，提供 `/health`、`/v1/models` 和 `/v1/chat/completions`，支持普通 JSON 与 SSE 流式响应：

```sh
tfllm serve /models/Qwen3-8B --served-model-name qwen3 \
  --context 4096 --max-num-seqs 4 --chat-template-kwargs '{"enable_thinking":false}'

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"你好，介绍一下自己"}],"max_tokens":128,"stream":true}'
```

`--host`、`--port` 配置监听地址；`--api-key` 或 `TFLLM_API_KEY` 设置 Bearer token。
模型默认服务名称为输入路径/仓库 ID 的最后一段，`--served-model-name` 可覆盖。
聊天模板来自模型，采用 Jinja 渲染；缺失时明确要求 `--chat-template FILE`。支持 `chat_template_kwargs`（例如 Qwen `enable_thinking`）、BOS/EOG、generation_config 中的 EOS、temperature/top_p/top_k/repetition_penalty、seed、frequency/presence penalty、max_tokens/max_completion_tokens、跨 token 的 stop 字符串和 `stream_options.include_usage`。
当前支持 `n=1`、system/user/assistant 文本消息，以及 user 消息中的 Qwen3-VL 图片 content parts，thinking 内容原样输出；不实现工具调用、结构化输出、视频/音频或独立 reasoning 字段。不支持的请求字段报错。
兼容客户端在普通聊天中附带的空工具字段：`tools: []/null` 配合省略、`null`、`"none"` 或 `"auto"` 的 `tool_choice` 可正常使用；显式 `tool_choice: "none"` 时工具定义不会进入模型提示。`parallel_tool_calls` 可为布尔值或 `null`，在工具禁用时不生效。非空工具配合自动选择、`"required"` 或指定函数仍明确报错，不能将纯文本输出当成工具调用成功。Chatbox 使用本服务时只启用 Vision，需要关闭工具调用/MCP/联网搜索；若旧历史含工具消息，应新建普通对话。

一个服务共享一份 TFLLM 模型、llama 模型实例和整芯片 NPU 调度器。decode 使用一个多序列 llama context、一个持久 CPU 线程池和按步合批的调度器；`--threads 32` 是整个 decode 的线程预算，2/4 路并发不会各自再创建 32 个计算线程。空闲线程睡眠，NPU prefill 的 pair 调度保持独立。
`--max-num-seqs`（默认 4）限制同时生成数量；每个槽位有独立 FP16 KV stream，与 TFLLM native prefill 共享地址。所有槽位的 KV 和公共 llama 工作区在启动时分配，普通内存占用会随最大并发数增加；物理 KV 容量向上对齐到 llama 的 256 token 边界，用户可见的 `--context` 上限不变。
空闲会话槽位按真实可复用前缀选择，同时检查图片身份与 RoPE 位置；仅当匹配前缀不少于该槽位将丢弃的后缀时，优先续用它。否则先用空槽，槽位用满后淘汰最近最少使用的空闲槽，避免 Chatbox 搜索判断等不同 system prompt 仅凭几个公共 token 就覆盖主对话。`TFLLM_REQUEST` 的 `session_slot`、`cache_selection`（`prefix/empty/lru`）、`cache_previous_tokens`、`cache_match_tokens` 可用于诊断；最终实际复用量仍看 `cached_tokens`。槽位有限，系统提示发生变化的请求不能复用变化位置之后的语言 KV。
已准备好的 decode token/CPU 短 prefill 在同一次 `llama_decode` 中提交，各自携带序列 ID 和位置、分别提取 logits。收集批次最多等待 200 μs，不等待仍在 prefill 的会话；单槽配置直接执行。当前固定版 llama 会把长度不同或编号不连续的序列拆为内部 ubatch，因此一次提交不保证只有一个计算图。启动日志应出现 `decode_scheduler=continuous_batch decode_threads=32`。
空闲槽位优先复用最长相同 token 前缀；HTTP 多轮请求仍需携带完整 messages，复用不依赖客户端会话 ID。`--max-pending-requests`（默认 64，包含正在执行的请求）限制准入，满时返回 429，等待槽位超过 60 秒也返回 429。
连接断开后在队列等待/解码边界取消；已提交的 NPU 工作完成后再释放资源。SIGINT/SIGTERM 停止服务时等待正在执行的请求退出。
`--context` 默认 4096，必须是 64 的倍数且不超过模型配置；prompt 加最大生成长度超限会报错，用户无需补齐 token。
每次请求打印 `TFLLM_REQUEST`，含 backend、native_tokens、llama_short_tokens、cached_tokens、prefill_ms、total_ms 和实际生成数；缓存命中后的短请求可能不提交新的 NPU 工作。
`--gpu-memory-utilization` 继续控制 NPU HugePage 池，不是整个进程的内存上限；llama 权重、CPU 工作区和各会话 FP16 KV 需另行计入。

`chat` 支持 `/clear` 清空聊天历史、`/exit` 退出，以及 `--system`、`--max-tokens`、`--temperature` 等参数。

本地验证：`ctest --test-dir BUILD -R 'tfllm-(chat-frontend|hf-import|vision-hf|llama-handoff|llama-batch|architectures)' --output-on-failure`。
chat 测试覆盖真实 GGUF/U8 导入、并发共享模型、缓存、文本 CLI、HTTP/SSE、鉴权、参数拒绝和取消；HF 测试现场生成小 Llama 及 Qwen3-VL BF16 safetensors 分片并调用真实固定版转换器，覆盖文本权重提取、缓存迁移、snapshot 配置变更与图像输入拒绝。架构测试使用真实 llama 比较 Qwen3-VL F32/F16/Q4_0/Q8_0 来源的 prefill/decode/追加/回滚，另检查长位置的 MRoPE 尾部。缺少可选 Python 依赖时相应 CTest 显示 SKIP，不算通过；这些测试均显式使用 CPU，不代表真实大模型 NPU 精度/性能验收。

## Dense mmproj 和图像输入

| 视觉结构 | GGUF `clip.projector_type` | 实现与导入范围 |
|---|---|---|
| MoonViT 2D | `kimivl` | 原生 patch Unfold、带抗锯齿的位置插值、二维 RoPE、补齐和 merger；导入配套 mmproj GGUF，提供 `VisionEncoder` 接口 |
| LocateAnything / MoonViT-SO + Qwen2 | `locateanything`（TFLLM 扩展） | HF `LocateAnythingForConditionalGeneration`；交错二维 RoPE、bicubic 位置插值、2×2 合并后 LN/MLP；支持官方 `slow` 自回归模式 |
| CLIP / SigLIP + MLP | `mlp`、`mlp_norm` | CLIP CLS/patch 选择、SigLIP 无 CLS、指定 hidden-state 层及拼接、projector；支持标准 HF LLaVA wrapper |
| InternVL | `internvl` | LayerScale、可选 Q/K norm、pixel shuffle v2、LN/MLP；支持 HF `InternVLChatModel` 和 `InternVLForConditionalGeneration` 的 dense decoder |
| Pixtral / Mistral Small 3.1 视觉 | `pixtral` | 二维 RoPE、SwiGLU、可选 patch merger、`IMG_BREAK`；HF `LlavaForConditionalGeneration`/`Mistral3ForConditionalGeneration` |
| Qwen2-VL | `qwen2vl_merger` | temporal patch=2 的静态图片展开、全局 attention、2×2 merger、语言 MRoPE；HF/GGUF |
| Qwen2.5-VL | `qwen2.5vl_merger` | 额外 RMSNorm、SwiGLU、窗口 attention 和周期性全局 attention；HF/GGUF |
| Qwen3-VL | `qwen3vl_merger` | 保留原有 DeepStack、位置插值和 interleaved MRoPE；HF/GGUF |
| SigLIP / Gemma3 | `gemma3` | SigLIP、局部平均合并、RMSNorm/投影、图像双向 attention 与语言滑窗；HF/GGUF |

MoonViT 支持指 **2D 视觉编码器**，不意味着支持 Kimi-VL 的 MoE 语言模型，也不包含 Kimi-K2.5 的 MoonViT3D/视频结构。需配合已支持且权重匹配的 dense decoder，不能把它任意接到另一个语言模型上。InternVL 当前使用最后一层特征和 pixel shuffle v2；v1 或其他选层配置在 HF 导入时明确拒绝。CLIP/SigLIP MLP 使用 patch-only 特征：CLIP 的 `default`、SigLIP 的 `full`。LLaVA-NeXT 的 `spatial-unpad`/可学习 `image_newline` 装配尚未接入，导入会报错，不会丢弃该权重继续执行。

LocateAnything 的 HF 导入同时生成原生语言权重包、专用 mmproj 和配套 llama decoder；BF16 来源仍自动生成 CPU Q4_0 decode GGUF。视觉 MatMul 和语言 prefill 使用配置的 NPU 后端，decode 使用共享 llama CPU 调度器。启动日志明确打印 `generation_mode=slow`：当前未实现官方 `fast/hybrid` 的六 token 并行框解码（PBD），不能据此比较官方 PBD 吞吐。输出保留 `<box>`、`<ref>` 和坐标特殊 token；可使用普通图文 OpenAI 请求，坐标仍采用模型自己的表示，服务不会自动转换为像素框。

该适配按 [官方模型配置](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/config.json) 和参考实现核对：MoonViT-SO 使用交错的 X/Y 旋转频率、无抗锯齿的 bicubic **位置嵌入**插值，projector 在合并后的 4H 上做 LayerNorm，并使用 erf GELU；这些与 `kimivl` 不同。图片使用 Pillow bicubic 缩放及 patch×2 向上对齐；`--max-image-tokens` 限制合并后的 token 数，对向上取整超预算的情况再缩小，可能略低于预算。只支持静态 RGB 图、dense Qwen2 和已合并的 LoRA 权重，不执行下载仓库中的 Python 代码。

```sh
tfllm serve /models/LocateAnything-3B --host 0.0.0.0 --port 8001 \
  --threads 32 --max-image-tokens 576
# 从 ModelScope 下载时明确选择来源：
tfllm serve LLMModel/LocateAnything-3B --model-source modelscope \
  --host 0.0.0.0 --port 8001 --threads 32 --max-image-tokens 576
```

`tfllm-locateanything-import` 用小型独立 Torch 模型检验真实 HF→GGUF→TFLLM 导入、非方形视觉特征、图文 logits、共享 KV 的 llama 接力 decode、多图/前缀复用、图像预处理和坐标词表。该测试在 host 执行且检查 NPU K/N 补齐路径；真实 3B checkpoint 的 NPU 量化精度、框定位质量和速度仍需板端验收。

```sh
# HF 自动提取视觉权重；已有量化 decoder GGUF 则指定配套的 mmproj。
tfllm serve /models/Qwen3-VL-4B-Instruct --max-image-tokens 1024
tfllm chat /models/Qwen3-VL-4B-Q4_K_M.gguf \
  --mmproj /models/mmproj-Qwen3-VL-4B-F16.gguf --image /images/example.jpg
# 逐算子日志同时包含 vision.unfold、视觉层、QK/AV、merger 和 decoder。
tfllm_profile chat /models/Qwen3-VL-4B-Instruct \
  --image /images/example.jpg --profile perf/vision
# 新增的 dense 模型使用相同入口。
tfllm serve /models/Qwen2.5-VL-3B-Instruct --gpu-memory-utilization 0.8
tfllm chat /models/InternVL3-2B --image /images/example.jpg
tfllm chat /models/pixtral-12b --image /images/example.jpg
```

`--image` 可重复，附到本地聊天的首个问题；后续轮次保留图像。HTTP 使用 OpenAI content parts：

```python
import base64, json, urllib.request
image = base64.b64encode(open("example.jpg", "rb").read()).decode()
body = {"model": "Qwen3-VL-4B-Instruct", "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
    {"type": "text", "text": "描述这张图片。"}]}], "max_tokens": 128}
request = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(request).read().decode())
```

目前接受 base64 data URL，远程 HTTP 图片 URL 明确拒绝；单张文件不超过 12 MiB、32 MP，每请求最多 16 张，JSON 请求上限 32 MiB。不把服务端文件路径当作远程 API 输入。Python 做 EXIF/RGB 解码，固定版 llama.cpp 的 mtmd 负责各模型的缩放、归一化、切片和图像 token/position 排列；归一化结果转为 FP16 CHW 交给 TFLLM 计算。mtmd 只加载张量形状，不分配第二份视觉权重，不执行其 CPU 视觉图。`--max-image-tokens` 默认 1024（范围 1..4096）；固定分辨率模型不能任意缩小到该值，超预算会拒绝请求。精度对照时需保持同样预处理配置。

上表结构均支持**静态图片**；视频目前仅支持下述 Qwen3-VL 适配，不包括音频、通用 CNN 或任意自定义 projector。Qwen 的 temporal patch=2 对静态图片重复两帧，空间 merge=2。未知 projector、未消费的额外张量、缺失权重、输出维度或 DeepStack 不匹配在启动时失败。GGUF F32/F16/BF16 和 ggml 可解码的块量化权重可读取，矩阵转成 FP16 后由现有 NPU 路径重新量化为 UINT8；归一化等向量权重保留 FP32。需要同一 checkpoint 配套的 decoder/mmproj，维度相同不能证明权重来自同一模型。

视觉和语言的静态 MatMul、动态 QK/AV 共用一个 `NpuScheduler`、HugePage 池和四个 pair 调度单元；NPU 模式不会把视觉矩阵乘回退到 CPU。CPU 负责 Unfold、LayerNorm、GELU、Softmax、RoPE 和残差等，采用 FP16 激活、FP32 归约/缩放；Unfold/LayerNorm 参考 TFDL，矩阵乘保持原有 UINT8/INT32 路径。注意中间激活精度与原 HF 浮点执行存在差别。视觉静态权重启动时准备，mmproj 本身仍为 GGUF，未增加离线硬件命令保存格式。

服务缓存最多 32 个图像切片、合计 128 MiB 的预处理输入及视觉特征，命中时复用编码结果；不同图片不会仅因占位 token 相同而复用 KV。KV 截断按物理 token 槽位，公开 `Session::Truncate` 拒绝截到一张图像内部。双向图像 attention 的一个 span 在同一 prefill chunk 内完成，内部静态 MatMul 仍按最多 1024 行分块；修改其中的位置或特征会回滚整个 span。图像特征缓存、llama 权重、CPU 工作区及共享 KV 不计入 NPU HugePage 池预算。

`tfllm-dense-vision-hf` 用独立 Transformers/Torch 对照各视觉结构，包含非方形 MoonViT/Pixtral/Qwen、位置插值、窗口 attention 和特征选层；每个模型另经 64 对齐的 blocked Linear 执行检查。`tfllm-dense-import` 从真实 HF 模型对象保存 safetensors，经 CLI 转换后测试图文 prefill、decode 和缓存，额外对照 Qwen2/2.5-VL 的图文 logits、接力 decode 及多图 MRoPE。`tfllm-vision-hf` 保留 Qwen3-VL 的视觉/DeepStack/logits、并发、多图和错误输入回归。`tfllm-fp16-test` 包含超过 1024 个图像 token 的双向 attention、回滚和滑窗边界。这些是 host 验证；真实 NPU 的量化精度、速度须在板端用实际 checkpoint 验收。

固定版 llama.cpp 的适配由 `apply_bridge.py` 幂等应用，包括 Qwen2-VL RMSNorm epsilon 读取、Qwen2 视觉 FFN 元数据、Pixtral `IMG_BREAK` 权重导出和 HF wrapper 命名兼容；重新运行 CMake 会检查固定提交并应用，HF 导入缓存随转换器变更失效。


## Qwen3-VL 视频输入

统一的 `/v1/chat/completions` 接受 `video_url` content part，支持 HTTP(S) 视频地址或 base64 `data:video/...`。这是 OpenAI-compatible 接口的扩展字段，客户端需要能原样发送它。当前原生视频适配为 **dense Qwen3-VL + 配套 qwen3vl_merger mmproj**；其他视觉模型的视频请求明确拒绝。导入格式和 NPU 权重包无需改动。

```sh
# 需安装新增视频解码依赖（requirements.txt 已包含）；板端安装目录：
python3 -m pip install -r /path/to/install/tfllm/share/tfllm/python/requirements.txt

tfllm serve /models/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000 \
  --threads 32 --max-image-tokens 576 --max-video-tokens 2048 \
  --video-max-frames 32 --video-fps 2

tfllm chat /models/Qwen3-VL-8B-Instruct --video /videos/example.mp4
```

```python
import base64, json, urllib.request
video = base64.b64encode(open("example.mp4", "rb").read()).decode()
body = {"model": "Qwen3-VL-8B-Instruct", "messages": [{"role": "user", "content": [
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + video}},
    {"type": "text", "text": "按时间顺序描述视频中发生了什么。"}]}], "max_tokens": 256}
# url 也可填写服务器能够访问的 https://.../example.mp4。
request = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(request).read().decode())
```

处理流程为公共文件解码/抽帧 → 模型视频适配 → 原生视觉编码 → TFLLM prefill → llama CPU decode。PyAV 在 CPU 上解码文件，只将选中帧转换为 RGB。Qwen3-VL 每两帧构成一个真实 temporal patch，奇数帧重复末帧；按两帧实际时间戳的均值写入 `<x.x seconds>`，再拼接对应 video token 和三轴位置。每对帧单独执行空间 attention，复用 NPU 视觉 MatMul/QK/AV 和 pair 调度，不同时保存整段视频的视觉中间激活。

参数含义：

- `--video-fps` 默认 2，范围 `(0,30]`，控制目标抽帧密度。`--video-max-frames` 默认 32、范围 2..64，超过上限时仍均匀覆盖整段视频，不截取前 32 帧；采用实际解码帧时间戳。抽帧策略会影响精度，不保证与每个上游 processor 的默认抽帧完全相同。
- `--max-video-tokens` 默认 2048、范围 1..4096，是**请求内所有视频合并后的视觉 token 总预算**，包括对话历史；多视频均分预算。每个 temporal pair 另受 `--max-image-tokens` 限制。缩放保留长宽比并取整到 patch×merge；不是每帧都分配 2048 token，也不是额外允许超过 `--context`。
- 每请求最多 4 个视频，每个文件最多 20 MiB、120 秒、源帧最多 32 MP；需有限时长、固定分辨率和有效时间戳。只处理画面，不处理音轨或直播流。下载和解码各有 30 秒边界检查；整个 JSON 上传仍限制 32 MiB。`--video` 仅用于本地 chat，远程 API 不读取服务器文件路径。

允许图片、视频和文本混合，保持 content part 顺序。有视频的 Qwen3-VL 请求统一由模型适配层预处理图片与帧；纯图片请求继续使用已有 mtmd 路径。预处理结果有 64 MiB 内容缓存，原生时序块/图片特征另有 128 MiB、最多 64 个条目的缓存。重复提问可复用视觉特征和 KV；改变帧内容、时间戳或位置会影响对应前缀匹配。缓存按容量淘汰，不保证任意长历史都命中。

`TFLLM_REQUEST` 输出 `video_pairs`、`video_tokens`。`tfllm_profile --profile` 的请求 CSV 包含原生视觉算子和语言执行；Python 文件下载、解码、归一化发生在 C++ 计时前，应结合客户端 TTFT 评估完整耗时。

`tfllm-vision-hf` 对照独立 Transformers 的真实双帧 patch、DeepStack、混合图像/视频 MRoPE、prefill logits 和 llama 接力 decode；还覆盖真实视频文件、HTTP/SSE、并发、CLI 和多轮缓存。`tfllm-video-frontend` 检查抽帧时间戳、末帧填充、总 token 预算及输入拒绝。测试使用本地小模型和 CPU 后端，真实 checkpoint 的 NPU 量化精度与速度仍需板端复测。

## 实际 NPU 的速度和精度对照

| 程序 | 是否使用真实 NPU | 验证内容 |
|---|---|---|
| `tfllm-llama-test` | 否，日志标记 `NPU_hardware_tested=0` | CPU 参考实现与 llama 的共享 KV、结构和多会话交接 |
| `tfllm-npu-lease-test` / `tfllm-benchmark-test` | 否，注入 SDK / host 测试 | 锁生命周期 / 测速统计与对齐流程 |
| `tfllm-npu-test` / `tfllm-npu-pool-test` | 是 | MatMul 数值、缓存可见性、真实并发与进程锁 |
| `tfllm-npu-llama-benchmark` | 是 | 完整 NPU prefill 对照源 GGUF 的 llama CPU，速度与 logits 误差 |

父项目启用 `TFLLM_WITH_LLAMA=ON` 和 NPU 后，重新 `make -j4 && make install` 即可得到新程序。不传模型参数时，自动生成两层、hidden=128、head_dim=64 的 Qwen3 合成模型，执行真实 NPU 的投影、FFN、QK/AV、LM head；适合先确认链路，不代表 Qwen3-8B 的性能或质量：

```sh
# 以下 SDK 指向父项目的安装前缀。
SDK=/workspaces/NPU40T/TFDL2_SDK
"$SDK/tfllm/test/tfllm-npu-llama-benchmark" \
  --lengths 64,128 --warmup 1 --iterations 3 > npu-llama-smoke.log 2>&1
```

真实模型使用已经转换好的原生包、与其内容标识匹配的源 GGUF，以及真实 prompt 的 token ID 文件。`prompt.ids` 是包含 chat template/BOS 等所需 token 的空白分隔整数；程序不自动补 BOS、不做文本分词。默认至少需 **1088 个 ID**（最长 prefill 1024 + 第二轮新增 64）；同一文件的前 64/128/256/512/1024 个 ID 用于长度扫描，不够时缩短 `--lengths` 或设 `--append 0`。

```sh
# 已有包可跳过转换；输出目录须预先存在。
"$SDK/tfllm/bin/tfllm-convert" /models/qwen3-8b.gguf /models/qwen3-8b --require-native
"$SDK/tfllm/test/tfllm-npu-llama-benchmark" \
  --package /models/qwen3-8b --gguf /models/qwen3-8b.gguf --tokens-file prompt.ids \
  --lengths 64,128,256,512,1024 --warmup 1 --iterations 3 --threads 8 \
  > npu-llama-qwen3-8b.log 2>&1
```

每个长度执行首次运行、额外 warmup、正式采样；每轮先清空两边 KV 再输入相同 prompt，避免 `Prefill(same_prompt)` 复用全部前缀形成虚假高吞吐。每轮随后用 llama 参考输出的 argmax 同时喂给两边做 4 次 decode，再追加相同的 64 个 prompt token；分别报告 `prefill`、`decode`、`append_prefill`。decode 本身走 llama CPU，误差反映两条历史 KV 的差异。元数据长度必须始终一致，NPU prefill 的实际 Linear 完成数必须增加，禁止静默 CPU 回退。

- `FIRST_RUN`：该长度首次调用，包含 JIT/上传；较早长度可能已留下部分计划，并非每行都是进程冷启动。
- `RESULT`：预热后墙钟平均、p50/p90、有效 token/s，以及 `speedup=llama_mean_ms/candidate_mean_ms`。候选耗时包含 NPU、CPU 算子、量化、缓存、反量化和仍然发生的 JIT；不包含模型/context 加载、KV 重置、参考执行和误差比较。动态 QK/AV 和 LRU 失配仍可能编译，不能称为纯 NPU busy time。首次加载/context 设置另有输出。
- `CHECK_LOGITS`：每个 prefill 末 token 和各 decode token 的**全词表**最大绝对误差、RMSE、NRMSE、余弦相似度、`KL(reference||candidate)`、top-1 ID 和 top-5 重合数。`RESULT` 汇总正式样本的最差误差和 top-1 一致率；这些是指定 token 序列上的对齐指标，不是整语料 perplexity 或模型质量验收。
- 默认只报告误差，结束为 `status=COMPLETE accuracy=reported_no_acceptance_limits`。可按模型验收标准传 `--max-abs` / `--min-cosine`，任一检查超限退出 2；非有限值、设备失败或路由不符退出 1。源 GGUF 与 UINT8 包之间的重新量化误差也会计入，不承诺逐 bit 对齐。
- `--verify-i32` 可抽样核对反量化前的原始 INT32，但会增加计时内工作，日志标记 `diagnostic_timing=1 speed_comparable=0`；定位数值错误后应去掉该开关测速。

这是单会话端到端延迟对照，两条路径顺序执行、按轮交替先后次序；参考 llama batch 上限 1024，候选 `--chunk` 默认 1024，两个 llama context 都使用 `--threads` 指定的 CPU 线程数。TFLLM CPU 算子使用至多 8 个参与线程的共享池。NPU 现在对每个底层 GEMM 使用整芯片 Grid 切分，再把两块一组的任务交给四个 pair worker；单请求可同时使用 8 核，多请求在分块边界公平分享。`--gpu-memory-utilization` 默认 0.8，控制整个引擎的 HugePage 预留；`--max-plans` 默认为 0，执行计划只受内存预算约束；正数设置额外数量上限（参数乘 4 为整芯片上限）。预留、静态权重转换与上传计入 setup；首次遇到新行数的命令编译仍计入 candidate。模型包、llama context/GGUF、FP16 KV 的普通 CPU 内存另行使用，不能把 HugePage 上限当成整个进程 RSS 上限。

## 逐算子性能日志与报告

板端重新 `make -j4 && make install` 后，使用带 `_profile` 后缀的 benchmark 并增加 `--profile 输出前缀`。先复跑同一合成模型，便于与之前日志比较：

```sh
SDK=/workspaces/NPU40T/TFDL2_SDK
mkdir -p perf
"$SDK/tfllm/test/tfllm-npu-llama-benchmark_profile" \
  --lengths 64,128 --warmup 1 --iterations 3 \
  --profile perf/qwen-fixture > perf/qwen-fixture.log 2>&1
```

生成三个文件：

- `.log`：原有速度/精度结果，以及每次算子的 `PROFILE_OP` 墙钟耗时、层名、形状、attention head/tile 行号。
- `.csv`：所有原始 span，包含父子 ID、线程、NPU pair、M/K/N、补齐后的 M、计划命中、descriptor 数、分配/输出字节数、淘汰数。每次 Append 计算结束后写盘。
- `.md`：端到端和精度对照、按成本排序的算子/层、各 NPU 阶段、逐算子阶段归属、计划命中/失配、队列/worker 汇总；分别统计 `first`、`warmup`、`measured`，显示调用均值、p50/p90、每次 Append 累计耗时和占比。

覆盖投影/FFN/LM head、Norm、RoPE 表生成/应用、残差、激活乘法、KV 写入/交接/历史整理、QK/Softmax/AV、gather/scatter。NPU Linear 分为 `plan_lookup`、`jit`、`input_quantize`、`cpu_cache_before`、`lease_wait`、`pair_prepare`、`npu_execute`、`pair_release_writeback`、`cpu_cache_dequant`；JIT 进一步记录 chip gate 等待、arena/淘汰、权重量化/打包、SDK 编译绑定。超长 K/N 的权重块打包和最终 narrowing 也有单独记录。NPU FP16 路径不再分配补齐用的 FP16 输入/输出中间缓冲。

K/V 预处理新增 `kv_host_prepare`（普通主存快照、完整 `[K][N]` 排布）与其子阶段 `kv_host_quantize`、`kv_host_transpose`。转置使用 32×32 分块、AArch64 NEON 8×8 字节内核，非 ARM 使用分块标量实现；不改变量化尺度、舍入或输出字节。执行槽换入记录 `kv_stage_copy`，`bytes` 是实际复制的量化字节数，`cache_hit=1` 表示槽内已有同一快照，本次跳过复制。`plan_weight_rebind` 现在位于 pair worker 内，只更新实际执行的大小核描述符；核类型迁移时再更新另一份。其子事件 `rebind_chip_gate_wait` 记录共享生命周期屏障的锁等待，`rebind_sdk_calls` 记录该 pair 实际修改的描述符（当前 ACP-only 实验版含命令修改与屏障，不执行 CPU cache clean）。后者的 `descriptors` 是调用数量。它们现在是可重叠的 worker-time，不能与旧版启动线程上的串行累计直接作为端到端收益比较。`plan_window_evict` 记录执行计划分配时的批量淘汰数，`weight_placement_evict` 记录新权重放置失败后的回收；这些指标仍只在 `_profile` 构建中存在。

**父 span 包含子 span，不能把不同表相加。多个 Attention head 或 pair 并行时，其算子/阶段累计是 worker-time，不能直接相加当作端到端时间，占比也可能超过 100%。** `npu_execute` 包含 launch/poll，是主机观测时间；`npu_submit` 已在其中。CPU cache + dequant 在同一 worker 内融合，报告它们的合计时间，包含调度/等待。字节数不是 DDR 流量计数，线程和 pair 时长也不是硬件利用率。报告中的 `total ms/Append` 可以比较一个算子在多头、多 tile、多 chunk 下的累计成本。

采集默认关闭。开启时会增加计时/内存记录开销，标记 `profiling=1 speed_comparable=0`；最终吞吐请用**不带 `_profile` 后缀的程序、相同参数去掉 `--profile`** 单独复测，且不要加 `--verify-i32`。新增采集文件的 I/O 不在 candidate 计时内；SDK 原有的 JIT 控制台打印仍计入 JIT。计时后的日志输出也可能影响下一轮频率/缓存状态。每次 Append 默认最多记录 100000 个事件，可用 `--profile-max-events` 调整。超限会明确输出 `TRUNCATED` 和 dropped 数，不能拿截断报告做完整成本占比分析。

Recorder 在请求执行前按事件上限预留数组容量，记录期间不再扩容或搬移已有事件，消除原来 65536/131072 事件附近的整表搬移停顿。上限越高，profile 预留的额外内存容量越大；计时、字符串和记录锁的开销仍存在。图文请求建议 `--profile-max-events 1000000`。

### 独立的 prefill 优化路径

公共接口保留 `RunFp16`；新增同步借用视图 `RunFp16Into`（行 stride、可选窗口行索引）、`QuantizedInput` 生产器和 `RunQuantizedInto`。CPU/第三方后端默认使用兼容路径；NPU 后端在 UINT8 执行空间补 K/M，INT32 反量化直接写有效 N 列和最终目标 stride。量化只扫描逻辑 K，填充值为零点 128；超长 K 继续在 FP32 中累加并最终缩窄一次。

| 文件 | 独立职责 |
| --- | --- |
| `include/tfllm/linear_view.h`、`src/linear_view.cpp` | 输入/输出视图、兼容实现、不可变量化输入快照 |
| `src/blocked_linear.cpp` | K/N 分块、权重补齐与长 K 累加，不包含模型特定融合 |
| `src/projection.cpp` | 分块投影、bias 和 FP16 边界 |
| `src/dense_ffn.cpp` | 无 gate 的 up → 激活 → down 路径 |
| `src/gated_ffn.cpp` | gate/up 共用一次输入量化；语言与视觉分别保留原激活舍入方式 |
| `src/vision_attention.cpp` | 非因果全局/窗口视觉 Attention、直接写出、Softmax→AV 输入融合和 head 流水 |
| `src/tiled_attention.cpp` | 语言 causal/SWA/KV Attention，独立于视觉路径 |

视觉融合仍把概率缩窄到 FP16（在寄存器中完成）后再量化，保持原 `Softmax → QuantizeRow` 的字节与 scale。支持并发的 NPU 后端默认最多两个 head 在途，进程内共用两个持久调度线程；这些线程独立于 CPU kernel 线程池，避免在 CPU pool 内等待 NPU。每个请求最多提交两个 head，按 FIFO 补充；底层继续以 pair 为调度单位。失败会等待所有已接受任务结束后再释放借用的输入/输出。Qwen3-VL 和其他 dense ViT 使用同一视觉 Attention 路径，预处理和各模型结构仍留在各自编码器中。

gate/up 的量化快照只在当前执行期间有效。若复用同一形状、同一高位窗口的执行槽，up 可直接复用 A；跨窗口或槽位则复制量化字节。任何普通输入写入都会使该缓存失效。没有引入额外静态权重副本，也没有把 gate/up/down 全部合成一个不可拆的执行分支；down 的整行 scale 归约和 FP16 边界仍单独执行。

新增 profile 事件：`shared_input_quantize`、`input_quantized_write`（`cache_hit=1` 时省掉再次写 A）、`vision_kv_gather`、`vision_qk`、`vision_softmax_quantize`、`vision_av_quantized`；通用后端使用 `vision_softmax_fp16` / `vision_av_fp16`。`ffn` 和 `vision_head` span 标识执行路径。head/worker 时长允许重叠，报告的 unassigned wall time 按算子区间并集计算。

此版新增 SDK `tfaccRebindGemmBDescriptor`，只读 A/B/C 的地址元数据，不改变共享 `MmapBuf` 的临时 offset。**SDK 和 TFLLM 必须一起重新编译安装**。描述符仍只允许在自身不执行时更新；分配/回收继续排斥执行与重绑定。

```sh
cmake --build build -j4
cmake --install build
SDK=/workspaces/NPU40T/TFDL2_SDK
"$SDK/tfllm/test/tfllm-linear-paths-test"
"$SDK/tfllm/test/tfllm-npu-pool-test" --verify-i32
"$SDK/tfllm/test/tfllm-cpu-benchmark" 30
```

第一个是 host 回归，覆盖逻辑尾块、索引视图、两种 FFN 路径、Softmax 量化逐字节对齐、并发与异常收尾；第二个在真实 NPU 上额外检查 M=17/K=73/N=77、有效列写出 guard、输入复用失效和融合 AV。CPU benchmark 同时报告 `softmax_av_separate` / `softmax_av_fused`，终点均为 UINT8+scale。端到端性能仍需以相同图片、token 数和预热状态在板上复测，最终速度使用普通构建。

### 全路径 ACP 实验与 B 地址重绑定专项测试

当前源码是 **ACP-only 实验版，重新编译 SDK 和 TFLLM 后直接生效，无需额外开关**。NPU40T SDK 和 TFLLM NPU adapter 的所有 CPU cache clean/invalidate 都已移除，包括首次/完整命令绑定、B-only 更新、地址表/LUT、prepared command、静态权重上传、动态 K/V 换入、输入激活与 INT32 输出读取。CPU/device 交接处保留 `dsb sy`；NPU 自身的 cache prepare/release/writeback、高位地址切换和进程锁继续执行。本版将 **数据读写也 coherent** 作为待板端验证的假设，v2 只免命令 clean 的 PASS 不能代替本版数据通路校验。

启动日志应同时出现 SDK 的 `NPU_CPU_CACHE mode=acp_only` 和 TFLLM 的 `TFLLM_CPU_CACHE mode=acp_only`，两者都有 `clean=0 invalidate=0`。普通版和 `_profile` 版使用同一策略；原 `cpu_cache_before` / `cpu_cache_dequant` 事件名保留便于对照 CSV，当前只包含屏障或屏障加转换，不再包含 CPU cache line 遍历。SDK 的全局修改也会影响链接这版 SDK 的 TFDL；TFDL 上层自行执行的 CPU cache 维护不在本次 TFLLM 实验范围内。无需修改或重载驱动。

`tfllm-npu-rebind-benchmark` v3 不需要模型，使用真实 SDK 分配的命令内存；覆盖语言小投影、大 FFN、视觉投影及 QK 形状。每个形状使用生产 Grid 策略切分 8 个 shard，并编译大小核两套描述符，测量整组 16 份描述符的绑定时间。分配、编译、真实 GEMM 校验均在绑定采样外，也不记录逐算子 profiler 事件。v3 不再提供 clean 基线；与保存的 v2 日志比较。

```sh
# 在现有板端构建目录重新构建、安装 SDK 和 TFLLM。
cmake --build build -j4
cmake --install build
SDK=/workspaces/NPU40T/TFDL2_SDK
# 先测试小命令/数据集，减少自然 cache 替换对一致性校验的掩盖。
"$SDK/tfllm/test/tfllm-npu-rebind-benchmark" --case acp_hot --verify-launches 100 --cpu 19
# 全部形状（含长 K）：
"$SDK/tfllm/test/tfllm-npu-rebind-benchmark" --cpu 19 > rebind-acp-only.log 2>&1
# 生产调度路径、跨层重绑定和动态 K/V：
"$SDK/tfllm/test/tfllm-npu-pool-test" --verify-i32
```

benchmark 运行时取得整芯片四对 NPU 的进程锁，退出时释放；应先暂停服务，避免与服务争抢。只申请一个 512 MiB arena，不预留 25 GiB 模型池。`--list` 不访问设备。日志输出实际加载的 SDK 路径；需要确认 SDK 的 ACP-only 启动标记，不能混用旧动态库。

| `REBIND_RESULT mode` | 测量内容（全部免 CPU clean/invalidate） |
| --- | --- |
| `same_address` | A/B/C 地址均不变，SDK 调用基线 |
| `full_A_B_acp_only` | 同时切换 A、B，强制完整绑定 |
| `B_acp_only` | 只换 B，更新命令和地址表，最后执行屏障 |
| `B_phased_probe` | 内部 helper 诊断：分别测量 patch/store 与 `dsb sy`，`clean_issue_us=0` |
| `B_acp_only_repeat` | 重复 B-only 路径，检查运行顺序和频率漂移 |

`REBIND_PLAN` / `REBIND_DESCRIPTOR` 输出命令总量、需要更新的 `B_patches`、`legacy_command_clean_visits` 与 `unique_command_lines`。`legacy_command_clean_visits` 仅描述旧版本的 clean 遍历次数，本版实际为零。重定位使用连续 `vector` 表，每次仍更新所有引用 B 的命令。`mean_group_us` 是整组描述符耗时，`mean_descriptor_us` 是除以描述符数的平均值。

探针每份描述符读取三次时钟，只在显式 benchmark 中执行；普通绑定没有增加计时或计数。修改阶段包含映射内存访问/cache miss；探针直接调用内部 helper，其他模式经过 SDK API，不能精确扣除 `same_address` 当作底层修改成本。

采样前比较完整绑定、B-only 与探针在两个 B 地址上生成的全部命令字，失败即停；此项 `command_equivalence=PASS` 仅检查 CPU 看到的字节。真实硬件校验默认每种场景 `--verify-launches 8`：

- `B_address_acp_only`：两份 B 内容为 X / -2X，先完整绑定旧 B，再更新 B 地址立即 launch；完整绑定也不 clean。
- `operand_reuse_acp_only`：保持 A/B 物理地址不变，交替改变 A 的符号及 B 的值，再次执行，检查新输入是否可见。

两种场景均只在开始时 poison C，之后复用 CPU 已读取的输出，检查 NPU 新写入是否可见；逐元素比较 INT32 和尾部 guard。每两轮交换大小核，覆盖 8 个物理核心和 16 份描述符。`REBIND_EXEC_CHECK ... status=PASS` 才代表本次板端数值通过；launch/poll 超时、NPU 写回失败或数值不匹配立即退出，不再回收设备缓冲。`--verify-launches` 必须为 4 的倍数，0 显式跳过硬件校验。

保留 v2 的 SDK 符号 `tfaccSetGemmBRebindCommandCacheClean` 以兼容已有调用方；在本实验版本中 true/false 都不执行 CPU clean，不能用它恢复旧策略。恢复原行为需要回退这次源码并重编 SDK、TFLLM。现有 `tfllm-npu-rebind-test` 验证命令/LUT 字节、长 K、批次、strided B、偏移与地址窗口；本地测试不能验证硬件一致性。最终是否更快需复测真实 prefill 和 TTFT，绑定微基准不包含 NPU 执行时间。

高位地址控制路径现在使用驱动初始化时建立的 ACP 寄存器映射，省去每次
`TF_SET_ADDRESS_HIGH` 在全局锁内的 `ioremap/iounmap`；HIGH 仍按任务更新，
pair 所有权、cache 写回和空闲释放规则保持不变。此优化需要在板端重新构建并
加载新的 `tfacc2.ko`，只更新 SDK 动态库不会启用。驱动加载日志应包含
`persistent address-high mappings chips=... pairs=...`。

排查偶发慢提交时，可另开诊断运行：驱动参数 `control_trace_us=1000` 输出
超过 1 ms 的 HIGH/解锁调用，区分全局锁等待与取得锁后的工作；
SDK 环境变量 `TFDL_NPU40T_CONTROL_TRACE_US=1000` 输出超过 1 ms 的 cache
ACK/配置调用，区分运行时互斥锁等待与硬件控制阶段。两者默认均为 0（关闭），
诊断日志不加入逐算子 CSV，开启后会影响时延。具体启用方式见
[驱动控制路径说明](../Driver/driver/driver/tfacc2/README.hugepage.md#高位寄存器常驻映射与慢调用诊断)。
新增 `tfllm-npu-address-test` 可在无硬件环境验证映射回滚、并发 HIGH 更新及所有权
检查，安装到 `tfllm/test/`；真实跨进程互斥和 cache 行为仍由板端测试验证。

pair cache 控制默认使用两核并行请求：prepare 先向两核发出 type=7 invalidate，
等待两核 ACK 后配置、开启 cache；release 先向两核发出 type=6 clean/invalidate，
等待两核 ACK 后配置、关闭 cache，再释放原有进程锁。两核的配置写入一起完成并
分别读回确认。整个过程只锁本 pair，任务调度粒度和完成后立即归还 NPU 的行为不变。
每次交接仍完整配置寄存器，不用本进程的缓存值推断外部 TFDL 留下的状态。
任一核超时都返回失败、保留进程锁；TFLLM 继续隔离故障芯片及保留相关缓冲。

这项优化只需重新构建、安装父项目的 `NPU40T` 库及 TFLLM，不需要再次修改驱动。
新 SDK 首次使用 pair cache 时输出 `NPU_PAIR_CACHE version=1 mode=parallel`。
环境变量 `TFDL_NPU40T_PAIR_CACHE_SERIAL=1` 可切回原来的逐核串行流程，启动时读取，
日志标记 `mode=serial`；用于同一二进制的 A/B 比较，不会跳过 cache 维护。
本轮保留原来的 yield 轮询方式，没有同时引入短自旋或跨任务持锁，便于归因。

```sh
# 先检查请求顺序、失败后的持锁行为以及真实板端数值/跨进程交接。
SDK=/workspaces/NPU40T/TFDL2_SDK
mkdir -p perf
"$SDK/tfllm/test/tfllm-npu-cache-test"
"$SDK/tfllm/test/tfllm-npu-command-test"
"$SDK/tfllm/test/tfllm-npu-pool-test" --verify-i32
TFDL_NPU40T_PAIR_CACHE_SERIAL=1 "$SDK/tfllm/test/tfllm-npu-pool-test" --verify-i32

# 相同模型/长度/线程/预热条件；分别保存日志，比较 prepare/release 及端到端。
TFDL_NPU40T_PAIR_CACHE_SERIAL=1 "$SDK/tfllm/test/tfllm-npu-llama-benchmark_profile" \
  --lengths 64,128 --warmup 5 --iterations 30 \
  --profile perf/cache-serial > perf/cache-serial.log 2>&1
TFDL_NPU40T_PAIR_CACHE_SERIAL=0 "$SDK/tfllm/test/tfllm-npu-llama-benchmark_profile" \
  --lengths 64,128 --warmup 5 --iterations 30 \
  --profile perf/cache-parallel > perf/cache-parallel.log 2>&1
```

可再按相反顺序各跑一次，排除频率/温度漂移；最终测速切换普通版并去掉 `--profile`。
`tfllm-npu-cache-test` 使用生产请求/等待/配置代码和注入寄存器，检查两种完成顺序、
立即完成、单核/双核超时、MMIO 发布/读回顺序和外部配置覆盖，不能验证硬件 ACK 时序。
`tfllm-npu-command-test` 另运行真实 `PairLease.cpp`，检查 prepare/release 失败不解锁，
成功清理后才归还两核；两项 host 测试均安装到 `tfllm/test/`。
慢调用诊断中的 pair 路径显示 `op=pair_cache_prepare|pair_cache_release`，`core` 为偶数核；
`mutex_wait_us` 是取得两核运行时锁的耗时，`work_unlock_us` 包含两核请求/等待/配置。
串行对照仍显示逐核 `cache_ack`/`cache_configure`，可用它进一步区分 ACK 与配置成本。

真实 Qwen3-8B 的采集用前面的 `--package / --gguf / --tokens-file` 参数加同一 `--profile` 即可；不传这三项仍然是合成模型。之前实测的基线和优先排查项见 [PrefillBaseline-20260914.md](Results/PrefillBaseline-20260914.md)。新增 profiling 的 host 回归为 `tfllm-profile-test`，注入 SDK 的阶段计时回归在 `tfllm-npu-lease-test_profile` 中；普通版 `tfllm-npu-lease-test` 仍执行数值和进程锁回归。

`serve/chat` 使用目录参数，每次启动创建独立 `run-*` 子目录，每个请求保存一份 `request-N.csv`，并发请求各有自己的 recorder：

```sh
"$SDK/tfllm/bin/tfllm_profile" serve /models/Qwen3-8B --profile perf/serve
"$SDK/tfllm/bin/tfllm_profile" chat /models/Qwen3-8B --profile perf/chat
```

记录在请求推理完成后写盘，包含算子名、父子 ID、线程、shape、pair 和各阶段时长；`--profile-max-events` 限制每个请求的事件数。CSV 的 `dropped` 列非零代表截断，服务请求日志也标明 `profile_status=TRUNCATED`。这里的请求 CSV 没有 benchmark 的 llama 精度对照/Markdown 汇总，保存成本会进入用户观察到的响应延迟，使用普通版做正式服务测速。`tfllm-run_profile` / `tfllm-run-concurrent_profile` 的 `--profile PREFIX` 则输出一份 `PREFIX.csv`。

连续 decode 新增 `llama_decode_queue_wait`（等待合批/CPU 的时间）和 `cpu_batch/llama_decode_batch`（执行加复制 logits）事件。后者 `M` 是本次提交的总 token 数，`K` 是参与的会话数，`N` 是词表大小，`head` 是该请求的会话槽号；`session_slot_wait` 记录进入生成槽位前的等待。一次共享批次会分别写入参与请求的 CSV，跨文件相加会重复计算，且这些子事件已包含在 `llama_short_decode` 中。普通版由宏完全移除这些采样代码。

SDK 调用方链接 `tfllm_profile`（NPU/llama adapter 同样使用 `_profile` target）后也可按请求开启采集，`SessionExecutor` 和 `LinearPool` 自动传递上下文：

```cpp
#include <tfllm/profile.h>
auto profile = std::make_shared<tfllm::ProfileRecorder>();
{
    tfllm::ProfileBinding capture({profile, 0, {}});
    auto logits = session.Append(token_ids);
}
auto events = profile->Events(); // 请求完成后读取；并发请求建议各用一个 recorder。
// 异步调用先提交，再等 future.get()，最后取 Events()。
```

这里的 capture 按线程/请求生效，关闭后不残留到 worker 的下一项任务。默认不开启时计时 scope 不读时钟、不加 recorder mutex、不分配事件记录。只对原生 FP16 路径细分算子；llama 的短 prefill/decode 记录为整个 CPU span，不伪装成 NPU 算子。

## 调用方式

```cpp
#include <tfllm/llama.h>
#include <tfllm/npu40t.h> // 只在启用 NPU40T 的构建中使用

auto model = tfllm::Model::Load("models/qwen3-8b");
auto llama = tfllm::CreateLlama("models/qwen3-8b.gguf", model->config, 4096, 8);
auto npu = tfllm::CreateNpuPrefill();
tfllm::Session session(model, llama.cache, npu, llama.decode, {}, llama.bridge);

auto logits = session.Prefill(first_prompt_token_ids);
logits = session.Append({generated_token});
// 完整 chat template 的 token 序列；会复用完全相同的已缓存前缀。
logits = session.Prefill(second_prompt_token_ids);
```

`Append` 输入仅为新增 token；`Prefill` 输入完整前缀，会比较 token ID 并截断分歧之后的 KV。调用方负责 tokenizer、chat template、采样及停止条件。返回最后一个真实 token 的 logits。只有实际执行过的 token 才计入 cache，刚采样但尚未 Append 的 token 不在 cache 中。

默认调度：本次块长 <=32 走 llama.cpp，其他走 TFLLM prefill；最大块长 1024，逻辑计算档位按 64 向上对齐。阈值可用 `Policy` 配置，尚未根据具体板卡测得最优值。NPU 内部 MatMul 可进一步按 16 行处理尾块。真实长度、档位和 cache 容量独立；补齐 token 永不提交到 KV。

一个 `Session` 内的调用串行执行。不同会话分别持有 KV；TFLLM 模型权重通过 `shared_ptr<const Model>` 共享。`LlamaModel(gguf, config)` 只加载/校验一次，通过其 `CreateContext(capacity, threads)` 创建共享 llama 权重的独立上下文；单独调用此 API 仍获得独占 context/线程池。chat/serve 改用 `CreateBatchContexts(capacity, sequences, threads)`，返回的各组 components 必须各绑定一个独占 Session；它们共享 context 和持久 decode 线程池，分别持有 KV stream。旧 `CreateLlama` 便利函数仍按调用创建独立模型。批量入口的 KV 按最大并发在启动时分配，公共 context 工作区另计。

## 多会话 prefill 与 NPU 成对调度

TFLLM 以整颗芯片为单位管理 NPU。`CreateNpuPrefill()`、`CreateNpuLinear()` 和 `CreateNpuPrefillPool()` 都默认把物理 `0–7` 核加入调度，内部仍按 `0–1 / 2–3 / 4–5 / 6–7` 四组执行。不再提供 core 数量或 pair 列表；多芯片机器只通过 `NpuOptions::chip` 选择芯片，例如 `chip=1` 使用 `8–15`。八核参与调度不等于引擎空闲时长期独占芯片。

每组有一个常驻 CPU 线程，在该线程上创建、使用、销毁 pair handler，遵守 SDK 的 core mutex 必须同线程加锁/解锁的要求。每个 MatMul 用 `MatMulSplitPolicy::Grid` 按整芯片切分，共享队列按就绪任务组公平分配 pair。空闲的组领取可执行分块，会话不永久绑定某一组。

原生 FP16 任务边界是一次 `Linear::RunFp16`：投影/FFN/LM head，或者一个 QK/AV tile。内部超长 K/N 分块依次提交底层 MatMul，每个 MatMul 的 Grid 分块组成一个调度任务组。计划查找和输入量化不持驱动锁；需要新编译时临时取得整芯片进程锁。已有命令的执行只取得完整 pair，执行 launch/poll、两个 core 的 cache writeback/invalidate 和 cache 关闭，立即解锁。CPU cache 维护和反量化在解锁后完成，只读写该任务自己的内存。命令/工作区及编译缓存保留到淘汰或销毁，计划回收也纳入下述芯片协调。几何尺寸过小时可能只有一个 core 执行命令，执行期间仍取得完整 pair，保护共享高位地址寄存器。

同进程、同芯片的原生 NPU 引擎共用 JIT/执行协调锁：编译、计划淘汰和析构使用独占锁，取得 pair 到完成缓存清理使用共享锁。已有计划的四组计算仍可并行，CPU 量化/反量化不占此锁。HugePage 启动预留和 SDK 命令编译还会临时取得四个 pair 的驱动进程锁，因为底层物理分配会对整颗芯片发出范围失效。固定池中的普通操作数分配只做用户态子分配；它不再触发物理池增长。外部 TFDL 若自行在未持锁时调用会影响整芯片的分配接口，仍需它或驱动侧协调，这不在 TFLLM 调度器控制范围内。

进程互斥使用与 TFDL 相同的驱动 `TF_APP_LOCK / TF_APP_UNLOCK` ioctl，加上 SDK 内的逐核 mutex。`tfaccTryLockNpuPair` 在任一成员忙时退回全部本次取得的锁；驱动尝试使用负超时参数，只尝试一次。没有新增一个 TFDL 不认识的文件锁。成功取得之后重新清除旧 cache、开启两个 core；释放后 SDK 忘记进程内缓存的地址高位，下次 launch 重新编程。

CPU 侧另用 `SessionExecutor` 排队整次 `Prefill`/`Append`，返回 future。同一 Session 按提交顺序执行，多个 Session 并行；每个 Session 自有 KV 和 llama context。CPU 会话线程做 Norm/RoPE/Softmax 时，不占用 NPU 执行组；短 prefill/decode 仍走各自的 llama CPU backend。输入量化由会话调用线程与共享 CPU 内核池协作，反量化由执行 pair 线程与该 CPU 池协作；pair worker 等待后处理结束后再领取下一组分块，尚未与本组下一次设备执行重叠。

```cpp
#include <tfllm/llama.h>
#include <tfllm/npu40t.h>

auto model = tfllm::Model::Load("models/qwen3-8b");
auto pool = tfllm::CreateNpuPrefillPool(); // 整芯片 0..7，不指定 core 数量

std::vector<std::shared_ptr<tfllm::Session>> sessions;
for (int i = 0; i < 4; ++i) {
    auto llama = tfllm::CreateLlama("models/qwen3-8b.gguf", model->config, 4096, 1);
    sessions.push_back(std::make_shared<tfllm::Session>(
        model, llama.cache, pool.backend, llama.decode,
        tfllm::Policy{32, 256, 64}, llama.bridge));
}
// 在 sessions/pool 之后声明；销毁时先完成已接收的请求。
tfllm::SessionExecutor executor({4, 64, {}});
auto first = executor.Prefill(sessions[0], first_prompt_token_ids);
auto other = executor.Prefill(sessions[1], other_prompt_token_ids);
auto next = executor.Append(sessions[0], next_token_ids); // 等 sessions[0] 前一请求结束
auto logits0 = first.get();
auto logits1 = other.get();
auto logits_next = next.get();
auto stats = pool.scheduler->Stats();
```

- `MatMulSplitPolicy::Grid` 与 TFDL `BuildMatMulGridSplit` 使用同一策略：尽量使用 8 个核，完全对齐时优先纯列切分，否则搜索行列组合。过小形状按对齐约束减少实际分块。
- 逻辑分块按每两个 shard 组成一个 pair task；同一任务在任意 pair 执行时使用对应的 16/8 MAU 描述符。单请求最多 4 个 pair；两个就绪请求各 2 个；三个为 2/1/1；四个各 1 个。更多请求轮流等待，不抢占已经启动的命令。按当前就绪 GEMM 分组分配，CPU 算子期间不会为该会话保留空闲 NPU。
- `SessionExecutorOptions::threads/max_pending` 默认 4/64；同一 Session 的请求仍按提交顺序执行。每个引擎同时最多保留 4 个独立执行实例，额外调用等待；`queue_capacity` 限制排队 pair task 数，默认 64。任务组原子入队；出错后仍等待组内所有兄弟任务返回，再释放输入/输出引用。
- 每个静态权重块只量化、上传一份完整 `[K][N]`，在全部行数和会话间共享。Grid 随 M 改变；B 的偏移为 `firstColumn`，行步长为完整 N，不复制分列权重。同一补齐 M/K/N 只缓存一份 CPU 编译模板，其中包含各 shard 的大核/小核版本；不同权重、HIGH 窗口与并发请求复用该模板。
- 每次 GEMM 从目标 HIGH 的运行分区申请独立 A/C、LUT、scratch 和临时 K/V，复制模板并按预先收集的重定位表修改地址。每个执行实例都有独立命令缓冲，绑定 B 不会改写 A 正在执行的命令。NPU 写回和 CPU 反量化全部结束后，运行内存立即归还子分配池；空闲模板不保留 NPU 激活或 K/V 工作区。持久 KV 保留在 llama/CPU；准备好的量化 K/V 主存快照在每次执行前拷入运行分区。
- NPU 视觉 Attention 按后端工作集选行块：最多 1024 行，补齐后的单个 QK score 矩阵最多 1048576 个元素（总 INT32 4 MiB，均分八核时每核 512 KiB）。1024-key、16-head 的一层从 128 次 QK/AV MatMul 降为 32 次；较大图像继续分块，每行仍使用完整可见 K/V。CPU/reference 默认维持原来最多 256 行。每个 MatMul 仍拆成 pair task，任务完成后释放进程锁，其他会话可继续调度，不持锁跨整个 head 或层。
- 引擎启动预留 HugePage，以及 direct command 缓存需要扩容时，临时取得整芯片进程锁；完成后释放。CPU 编译模板、固定池子分配、命令缓存命中时的绑定/归还可与其他 pair 执行并行。命令缓存按需增长，每池最多 256 MiB；独立缓冲归还后复用，不随历史形状无限增长。
- 已绑定任务只取得其执行 pair 的锁，完成 writeback/关闭缓存后立即归还。闲置、CPU decode 不持驱动锁。外部占用重试到 `lock_timeout_ms`（默认 30000）。真正的物理扩容与执行由同进程 chip gate 隔开；执行/缓存准备/写回失败后保留相关实例、物理内存与未释放的锁，将整个 chip 标记不可用，不自动重放。Session 的块级 KV 回滚规则保留。
- `worker_cpus` 可指定四个 pair worker 的 Linux CPU；`SessionExecutorOptions::cpus` 指定会话线程 CPU。`Stats().logical_completed` 为逻辑 Linear 数；`workers[].completed` 为 pair task 数。执行/等待均是主机墙钟，不是硬件 busy counter。

### 启动内存预算

`NpuOptions::gpu_memory_utilization` 和 CLI 的 `--gpu-memory-utilization` 范围为 `(0,1]`，默认 0.8。按目标 chip 注册的 HugePage 总容量计算，向下取整到完整 1 GiB 单元：32 GiB × 0.8 → **25 GiB**。不足 1 GiB、没有 HugePage、空闲块不足均明确报错，不回退到额外 DDR 或悄悄提高上限。

通过新的 SDK 私有池 API 一次预留，在多个 4 GiB 地址窗口中管理；25 GiB 连续地址空间通常对应 7 个窗口。TFDL 原有可扩容 arena 不受影响，也不能分配这块 TFLLM 私有池。模型初始化检查非 KV 权重/workspace 下限，实际 NPU 分配按对齐字节计费。每个 HIGH 窗口的高地址端保留运行区，其余低地址空间放静态权重；当前默认运行区 1 GiB、权重区 3 GiB，严格按实际预留段裁剪，两区不能互相借用。静态权重驻留，运行数据在执行后归还。空闲模板可按 CPU 模板数量上限淘汰。

1 GiB 是注册页和预算取整的单位，`TF_MODEL_POOL_ALLOC` 返回的连续段可以更小。驱动在窗口低地址零处保留 1 MiB，IOVA 也可能跨窗口；预留按实际返回字节累计，在窗口剩余不足 1 GiB 时缩小申请，直到凑齐预算。不能要求每段恰好 1 GiB，否则连续 32 GiB 的八个窗口可能最多只收集到 24 GiB。`--gpu-memory-utilization 1` 也可能因保护区或已有占用而失败；不会降低实际预算来假装成功。

预留前打印 `NPU_MEMORY_REQUEST`；失败信息保留 requested/reserved/registered/eligible 字节数、最后一个 HIGH、驱动操作及 errno。部分预留失败时回收已获得的所有段。此修复不改变驱动 ABI，重编 SDK 与 TFLLM 后使用原命令重跑即可。

环境变量 `TFLLM_NPU_RUNTIME_MIB` 配置**每个 4 GiB 窗口的运行区大小**，单位为整数 MiB，当前默认 `1024`，允许 `512..4095`。空值、非整数或越界值在创建池时明确报错；小于 4 GiB 是为了保留权重区。配置在创建池时读取一次，更改已有进程环境不会移动现有池的边界，需重建引擎。

| `TFLLM_NPU_RUNTIME_MIB` | 每窗口权重区 | 每窗口运行区 |
| --- | --- | --- |
| `512` | 3.5 GiB | 512 MiB |
| `1024`（当前默认） | 3 GiB | 1 GiB |
| `2048` | 2 GiB | 2 GiB |

```sh
export TFLLM_NPU_RUNTIME_MIB=512
# 然后使用原有 tfllm serve 命令，--gpu-memory-utilization 仍控制总预留量。
```

设运行区大小为 R，布局为 `[0, 4 GiB-R)` 静态权重、`[4 GiB-R, 4 GiB)` 运行槽。这个变量与 `--gpu-memory-utilization` 独立：**总预留量相同，调小 R 可以把更多窗口内地址分配给权重**，避免权重被固定 2 GiB 分区卡住。它不自动减少或增加 HugePage 总预留量。

运行区由一个引擎同时在用的最多 4 个实例共享，不是每实例另有 R。保护区、空洞、非对齐 IOVA 和不完整窗口会减少实际容量。启动 `NPU_MEMORY` 打印 `weight_partition_MiB/runtime_partition_MiB`；SDK `NPU_MEMORY_LAYOUT` 打印名义分区大小，`NPU_MEMORY_WINDOW weight_capacity_bytes/runtime_capacity_bytes` 打印每窗口实际容量。运行区为 2 GiB 时，25 GiB 连续预留通常只有约 12 GiB 可用于配有运行区的权重窗口；调小 R 后按实际窗口日志计算。没有足够运行区的尾部窗口不用于静态权重放置。

静态权重按一次 K×N 分配及 16 KiB 对齐计算，不均匀 Grid 和新 M 不增加权重副本。一次 GEMM 的所有操作数与内部 LUT/scratch 必须在同一 HIGH，运行槽选择静态 B 所在窗口；动态 K/V 从有空闲容量的运行区申请。某窗口暂时不足时，在生成命令之前回滚部分租用，再等待在途任务归还；没有在用实例仍放不下时明确报错。全池预算、分区容量和最大连续段都需要满足，不依靠把权重挤入运行区解决。

命令队列受硬件 direct-fetch 限制，仍使用 reserved-DDR，独立于上述操作数布局。池内命令缓冲按 16 KiB 对齐复用，物理缓存默认每次至少增长 4 MiB、总计最多 256 MiB，整个缓存容量计入同一非 KV 预算。空闲时运行区租用归零；整池 `used_bytes` 仍包含静态权重与匿名 direct 命令池的物理预留量。FP16 持久 KV 仍由 llama/CPU 分配；该预算不限制持久 KV 或全部进程内存。

Session 初始化转换、打包并上传静态矩阵；`Prepare(1024)` 不预编译 1024 行命令。首个 M/K/N 编译不可变 CPU 配方，包含命令图像、相对布局和地址补丁。每次执行先租用一块完整操作数工作区与一块独立 direct 命令区，再把命令直接写入最终缓冲并修正 A/B/C/HIGH。内部 LUT/scratch 和输入输出只是偏移视图，不逐项分配，不克隆 `MatMulHelper`。全部 pair 写回和 CPU 读取完成后归还两块空间。

**空闲实例池、LRU 和 `TFLLM_NPU_IDLE_INSTANCES` 已删除。** 最多 4 个准备/在用任务共享运行区；每个同时执行的任务仍持有独立可修改命令、A/C 和临时 K/V。CPU 配方缓存不保留运行时物理地址。编译、打包、命令生成在各查询/申请锁之外；真实物理扩容仍独占 chip gate。详细生命周期、主机测试与板端验收见 [NPU_RUNTIME_SLOTS.md](NPU_RUNTIME_SLOTS.md)。

新增 `tfaccGetGemmExecutionSize`、`tfaccCreateGemmExecution`、`tfaccMemoryView`、`tfaccCommandMemAlloc`；必须一起重编安装 NPU40T SDK 和 TFLLM，驱动 ABI 未变。SDK 旧 GEMM/clone API 仍保留，TFLLM 不再使用 clone。

```sh
# 每个进程通常只创建一个引擎，让所有 Session 共享。
./tfllm-npu-llama-benchmark_profile --gpu-memory-utilization 0.8 --profile /tmp/tfllm-chip
# 普通/并发入口也接受此选项，放在模型参数后的 token 列表中：
./tfllm-run PACKAGE SOURCE.gguf --gpu-memory-utilization 0.8 TOKEN_ID...
./tfllm-run-concurrent PACKAGE SOURCE.gguf 4 --gpu-memory-utilization 0.8 TOKEN_ID...
```

日志 `NPU_MEMORY` 打印注册容量、比例、预留容量；`NPU_MODEL_PREPARED` 打印初始化后实际用量。SDK `tfaccModelMemoryPoolUsed` 的统计包含对齐后的操作数、内部临时量及整个 direct 命令缓存容量。更改了 SDK API，板端必须一起重新构建/安装 `libNPU40T` 和 TFLLM，不能只换测试可执行文件。

板端可先跑可核对数值的矩阵压力测试，再跑实际模型多会话示例：

```sh
./build/tfllm-board/tfllm-npu-pool-test
# 数值定位：在反量化前抽样检查原始 INT32，失败报告实际 pair/core、坐标和 INT32 参考。
# 诊断会增加 CPU 工作和缓存维护，不能用它比较性能。
./build/tfllm-board/tfllm-npu-pool-test --verify-i32
# 对照：保留四种形状的计划，或只使用一个 CPU 调用者；仍使用整芯片调度。
./build/tfllm-board/tfllm-npu-pool-test --verify-i32 --max-plans 8
./build/tfllm-board/tfllm-npu-pool-test --verify-i32 --callers 1
# 不指定 NPU 数量，默认整芯片。测试期间除它启动的探测子进程外应无其他设备任务。
# prompt_tokens.txt 为同一个完整 prompt 的空白分隔 token ID，长度 >32。
# 四个独立 Session 并发 prefill，然后各执行一次 llama decode。
xargs ./build/tfllm-board/tfllm-run-concurrent models/qwen3-8b source.gguf 4 < prompt_tokens.txt
```

压力测试用 8 个 CPU 调用线程提交 96 次矩阵乘，交错权重、输入、M 形状和 K=128/4160，检查 INT32 参考窄化后的 FP16 结果与任务计数。它还用 `posix_spawn` 启动独立探测进程：引擎初始空闲、执行结束且计划仍缓存时应能拿到所有 pair；父进程持锁时子进程必须得到 BUSY。空闲重复创建/销毁引擎也纳入检查。多会话示例也共享 llama decode context/4 线程池；其 prefill 时间包含首次 JIT，不含模型/context 创建和随后的 decode；它打印总体 tokens/s 及每组分配统计。该并发路径尚未上板验证吞吐提升。

`tfllm-npu-command-test` 在 ARM 主机上使用真实 SDK Executor/MmapBuf、注入底层驱动，检查并发淘汰高地址命令缓冲时不会触发整芯片操作数 cache invalidation，并确认缓冲正常回收、普通 legacy 操作数仍保留原有清理。该测试随 `TFLLM_BUILD_TESTS` 构建并安装到 `tfllm/test`；它补充 `tfllm-npu-lease-test` 的 SDK API 模拟，不替代上面的真实 NPU 压力测试。修复 SDK 释放路径后应重新构建、安装父项目的 `NPU40T` 库以及 TFLLM，不能只替换测试可执行文件。压力测试数值失败会打印形状、行列位置、INT32 参考及预期/实际 FP16 位模式，量化尺度均为 1，不放宽比较容差。

## KV 的所有权与接力

未接 llama 时，`KvCache` 自己分配内存。接 llama 时，**llama.cpp 分配 FP16 K/V，TFLLM 直接引用这些 buffer 的地址与 stride**，不通过保存/恢复 session 文件搬运 KV。引用拥有 context 的生命周期，避免 context 先释放。当前限定 CPU FP16 KV、单 sequence 0、连续位置、无 cache shift/MLA/量化 KV；拒绝不兼容布局。Gemma 2/3 支持 llama 的 dense/SWA 双 cache，按全局层号映射，并分别发布/校验元数据。设置 `swa_full=true` 保留整个会话容量的 KV，因此能回退到较早前缀；这里没有使用只保留窗口长度的环形缓存节省内存。

K 存储 Q/K norm 和 RoPE 之后的值；按 KV head 数保存，不复制成 query head 数。V 支持当前 llama 非 Flash Attention 的转置布局，也检查非转置布局。KV 的物理 token 槽位等于绝对 token 位置。

每个块遵循：

1. 同步前一引擎，验证两侧元数据中的历史长度一致。
2. 只写 `[past, past + real_length)`；Attention 第 r 行的因果上界为 `past+r+1`；滑窗层再限制对应历史窗口。
3. 等待全部计算/写入完成，验证 logits，再发布 token/position/sequence 元数据。
4. 失败时撤销本块元数据；先前完整提交的块仍然有效。中途失败的长请求可能已提交前几个块，调用方可用 `Tokens()` 检查。

llama decode 自己写入和发布其 KV；TFLLM prefill 则通过桥接器发布 llama 的 cell 元数据。只共享一个内存指针、却不更新这些元数据，是不够的。

FP16 KV 位于 CPU 可访问内存；NPU 实际读的是独立 UINT8 执行缓冲，不是直接 DMA 读取 llama 的 malloc buffer。语言和视觉 attention 每次处理一个 KV head 时，先通过 `PrepareDynamic` 在普通主存生成不可变量化快照，提前准备完整 query 块和尾块需要的转置/列切分排布。K/N 需要补齐或分段时，同样提前准备每个 block；数值量化边界沿用原路径。

换入执行空间只复制已准备的字节和尺度、完成 CPU cache 维护，不做量化或矩阵转置。槽内已有同一快照时连复制也跳过；并发会话使用各自快照及独占的在用槽，NPU 完成、cache 写回及 CPU 读取完成后槽才可复用。主存快照是运行时对象，不写入模型包。

这版快照供一次 attention 的各个 query/GQA 块复用；追加 token 后仍重新准备有效历史。V 仍按输出 channel 沿历史长度计算 scale，未引入 per-token V scale 前移到 Softmax 输出的数值变化。因此当前不是永久增量 INT8 KV，也不是 UINT8 KV 零复制；llama decode 继续直接使用原来的 FP16 KV。

## FP16 激活与 CPU 内核

原生 prefill 默认使用 `CreateFp16Prefill`：embedding、残差、Norm 输出、Q/K/V、RoPE 输出、QK scores、Softmax 概率、AV、FFN 和 LM head 输出均为 `Activation = vector<uint16_t>`，存储真实 IEEE FP16 位。`CreateNpuPrefill` 自动接入这条路径。`RunFp16` 从 Session 一直贯通 LinearPool、K/N 分块和 NPU adapter，普通路径没有整块 FP32 激活往返转换。

| 数据/阶段 | 类型 |
|---|---|
| 算子间激活与共享 KV | FP16，单元素 2 字节 |
| NPU 工作输入 / MatMul 累加 | UINT8 / INT32 |
| Projection BiasAdd | FP16 bias + FP16 activation，FP16 输出 |
| Norm、LayerNorm beta、DeltaNet 参数、RoPE 因子、量化 scale | FP32 小张量或系数 |
| Norm 归约、exp/Softmax 归约、RoPE 运算、反量化 | FP32 内部计算，FP16 输出 |
| K>16384 的跨命令部分和 | 算子内部 FP32 累加缓冲，整个 K 完成后一次窄化 |
| 公共 logits API | 最后一行 FP16 head 输出拓宽为 float，softcap 与采样接口保持 float |

FP16 不会扩大数值范围：没有将溢出静默钳成 65504。NPU INT32 先按 FP32 scale 反量化，再窄化；长 K 也不先将各个部分和转 half，否则即使最终结果可表示，正负部分和也可能先溢出。

CPU 热点按当前 TFDL 实现的做法整理为独立内核，不引入 TFContext/图执行器依赖：

- 量化参考 `TFEngine/Op/Quantize.cpp`：FP16 绝对值位序求最大值，NEON 向量缩放/舍入/打包，逐行 scale；F16 K/V 工作权重也走该路径。
- 反量化参考 `TFEngine/Op/DeQuantize.cpp`：INT32 转 FP32、乘行列 scale、直接窄化，N 分片直接散写完整 FP16 输出。每个 CPU 分段先维护自己的 INT32 cache 行，紧接着转换；NPU 锁已释放。
- RMSNorm 使用 `LayerNorm.cpp` 的 FP16 load/store、多个 FP32 累加器与按行并行方式，但保留 RMSNorm 的均方根公式。
- Softmax 使用 `internal/SoftmaxEval.h` 的稳定最大值归约、向量 exp、FP32 求和与 FP16 输出；exp 多项式来自 `Common/Utils.h` 的 Cephes 实现。FP32 scratch 限于每个 CPU 任务的一行，不是完整 scores 张量。
- SiLU/GELU 与乘 up 融合，一次输出 FP16；残差、bias、embedding 转换均有 NEON 路径。GELU 保持各原生模型要求的 tanh 近似公式。
- RoPE 的 sin/cos 每个位置/维度计算一次，供 Q/K、所有 heads 和相同配置的层复用；旋转使用 NEON，支持 NEOX/交错布局。
- KV 写入和历史读取直接复制 FP16 位；连续行走 memcpy，V 转置按小块处理，两种 llama stride 布局均不经过逐元素 FP32 中转。

CPU 内核共享一个常驻池，最多 7 个后台线程，加上当前调用线程；小任务直接执行。不同会话/不同 NPU pair 共用这些后台线程，不为每个算子创建线程。池在 NPU worker 绑定 CPU 前初始化；后台线程继承首次初始化调用者的 affinity，不要在创建引擎前把主线程限制到单个 CPU。线程池回调只运行 CPU 内核，不能在其中等待 NPU 或嵌套并行任务。

`ReferenceBackend` 和 F32 `Linear::Run` 仍保留作数值对照。第三方 Linear 未实现 `RunFp16` 时默认走显式兼容桥；生产 NPU adapter 和分块/队列均实现原生 FP16，不走这个桥。未启用 NPU 的 `CreateFp16Prefill()` 使用 CPU dot 参考实现，不是用于大模型性能的 GEMM 库。llama.cpp 的短 prefill/decode 仍采用 upstream 自己的混合精度计算，本次没有改写其算子精度。

实现：[FP16 执行路径](src/fp16_prefill.cpp)、[CPU 内核](src/cpu_kernels.cpp)、[FP16 API](include/tfllm/activation.h)。本地微基准与板端复现方式见 [FP16 CPU 测试记录](Results/FP16Cpu-20260911.md)。

## 模型包

[schema/model.fbs](schema/model.fbs) 当前写出版本 2，并兼容读取版本 1 的 Qwen3 包。新增执行路径、总参数量、原始 GGUF 路径、RoPE、激活/缩放/softcap、逐层滑窗配置：

| 文件 | 内容 |
|---|---|
| `PREFIX.tfllm` | FlatBuffers 配置、权重目录、workspace 要求、命令计划及重定位元数据 |
| `PREFIX.weights` | 64 字节对齐的 F32/F16 或逐输出通道对称 UINT8 权重与 FP32 scales |
| `PREFIX.commands` | BlasOp 命令模板及配套 LUT/地址状态镜像 |

大 payload 使用外部文件与 64 位偏移，避免将 8B 模型塞进 FlatBuffers 的 32 位内部寻址空间。权重只读 mmap，不为每个 Session 复制。writer 按 tensor 流式写，完成前校验整个包；已有目标文件不会被静默覆盖。格式需 little-endian 主机。

```sh
# 输入 GGUF 可以是浮点或 ggml 已支持的量化权重。
# 原生路径默认将矩阵转换为逐输出通道 UINT8；embedding 和 projection BiasAdd 为 FP16，norm/LayerNorm beta/RoPE 因子为 FP32。
./build/tfllm-llama/tfllm-convert source.gguf models/qwen3-8b
./build/tfllm-llama/tfllm-inspect models/qwen3-8b
# --f32 用于同权重数值对照，避免再次权重量化。
./build/tfllm-llama/tfllm-convert source.gguf models/qwen3-reference --f32
# 必须生成原生包时启用严格模式，遇到需回退的结构直接报错。
./build/tfllm-llama/tfllm-convert source.gguf models/native-only --require-native
```

输出父目录需预先存在。转换器每次处理一个 tensor，不解量化整个模型后再复制一次。源 GGUF 的内容标识保存在包里；接 llama 时重新检查，防止误接另一个相同维度的模型。FNV-1a 标识用于意外错配检测，不是密码学签名。

支持单文件和官方 `-00001-of-000NN.gguf` 命名的分片文件；必须传入第一个分片。所有分片均参与参数计数和内容标识。转换时关闭 llama CPU 权重重排，从原始 ggml block 解码；运行时仍允许 llama 使用自己的重排优化。支持的输入量化类型取决于固定 ggml 版本的 `to_float` 解码器，当前实际回归了 F32/F16/Q4_0/Q8_0；没有逐一验收所有 IQ/K-quant 变体。

若选择回退，写出 `execution=llama` 的配置清单，`.weights`/`.commands` 为空，原始 GGUF 是必需运行资产，不会伪造 NPU 编译结果。`tfllm-inspect` 和转换器会打印回退原因。部署时须保留原始 GGUF 的全部分片，`tfllm-run` 的第二参数可以指定部署后的源路径，内容标识必须一致。

回退 API 使用 `GgufSession(source, capacity, threads, expected_identity)`，提供 `Append/Prefill/Tokens`，全部计算及 KV 都由 llama 管理，不与原生 prefill 混用。若模型的 memory 不支持部分截断，会清空重放完整前缀；解码执行异常时清空该 session，避免继续复用不确定状态。

默认 UINT8 转换会带来新量化误差；从 Q4 等 GGUF 解码再量化也不能恢复原模型精度。结构与小模型回归不能代替实际模型精度验收。

## NPU40T

仅链接本 checkout 编译的 NPU40T 后端库及其底层类型，不经过 `TFExecutor`。须先完整重编后端库，包含新增的 pair-lease API、prepared-command API 和线程局部 `ErrorInfo`；后者避免并发绑定时读写同一个错误字符串，不能混用新头文件和旧 SDK 目标文件。旧版 `libNPU40T.so` 也没有这些新增 API 的导出符号。在已经配置好板端工具链的环境中：

```sh
cmake -S TFLLM -B build/tfllm-board \
  -DCMAKE_BUILD_TYPE=Release -DTFLLM_WITH_LLAMA=ON -DTFLLM_WITH_NPU40T=ON \
  -DTFLLM_NPU40T_LIBRARY=/absolute/path/to/libNPU40T.so \
  -DTFLLM_NPU40T_CONFIG_INCLUDE=/absolute/path/to/configured/board/build
cmake --build build/tfllm-board -j4
./build/tfllm-board/tfllm-npu-test
# 各入口默认管理 0..7，芯片可用 NpuOptions::chip 选择。
./build/tfllm-board/tfllm-run models/qwen3-8b source.gguf 1 2 3 4
```

`tfllm-run` 的最后参数是 token ID，不是字符串。原生包在启用 NPU 的构建中使用 NPU prefill；未启用时使用 CPU reference prefill，后者不适合大模型性能使用。回退包无论是否启用 NPU 都使用 `GgufSession` 的整模型 CPU 路径。四个 token 的示例只触发短路径；测试 NPU prefill 需传入超过 `short_threshold` 的 token。

实现范围：

- 投影/FFN/LM head：运行时逐行量化 A，直接调用 backend 编译 UINT8×UINT8→INT32，直接反量化为 FP16。
- 多核：整芯片四个 pair 参与任务调度，以整芯片八核粒度复用 SDK 的 `MatMulSplitPolicy::Grid`，优先完全对齐的列切分，再考虑二维切分。静态 B 仅保存一份完整矩阵，各 shard 通过列偏移和原矩阵行步长读取。
- Attention：按 KV head 组织 GQA，分块执行 QK、带 `past` 偏移及模型滑窗/softcap 的稳定 softmax、AV；不漏掉模型要求的 key。QK tile 的逻辑 INT32 预算 <=1 MiB；每个 tile 的 NPU 写回/CPU cache 维护完成后再消费。
- 大矩阵：`CreateBlockedLinear` 将 K/N 超过 16384 的权重分块，尾部 K/N 补零到 64。K 分块的部分结果各自完成反量化后用 FP32 累加，N 分块散写回完整输出；bias 在完整 linear 结果上只加一次。UINT8 权重保留原输出通道 scale，输入逐块重新量化，因此误差可能与单块计算不同。
- 单次底层 MatMul 仍要求 M<=1024、K/N 对齐 64、K<=16384。上层可处理更宽的 FFN 和大词表 LM head；没有丢弃 K 项或裁剪模型通道。NPU 路径的静态 host 分块权重保留到引擎析构，NPU 打包权重在行数/会话间共享；这些 host 数据及 llama 权重/KV 仍占普通 CPU 内存。
- 本版 tiled Attention 上下文最多 16384，超限报错；不因模型支持 128K 就宣称已实现 NPU 128K。
- 静态权重缓存只区分权重身份，与 M 和 Grid 列布局无关，静态执行计划按窗口和形状复用；动态操作数另用私有槽及主存快照。整芯片非 KV 预算固定，内存不足时批量回收空闲计划，已选定窗口的操作数分配只回收该窗口，也可显式设置 LRU 数量上限；正在使用的计划不可淘汰，物理池不增长。
- 每个 pair 先 launch 自己的两个 shard 再轮询；四个 pair 可并行。完成 writeback 后进行 CPU cache 维护/反量化。NPU 超时/执行错误后保留相关物理内存及 core 锁，禁用该 backend，要求进程/设备恢复；不会在不确定 DMA 完成时释放内存。

该版本重视打通运行链路，CPU 内核已加入 NEON、常驻线程与 FP16 存储；实际板端吞吐尚未测量。计划数量上限过小会导致重复编译；静态打包权重仍保持驻留。它目前不是性能优于原 TFDL 的已验证结论。

## 已存命令的动态地址更新

[PreparedDescriptor](backends/npu40t/prepared_descriptor.h) 通过新增的 `tfaccCreatePreparedCommandDescriptor` / `tfaccSetPreparedCommandDescriptor` 持有 backend 内部分配的 `TFACCOpDescriptor` 子类，避免客户端链接后端隐藏的内存管理符号。它从 `Program` 装载命令与辅助镜像，通过 `Rebind()` 更新输入、输出、权重及 workspace/KV 地址，然后用已有 `tfaccOpLaunch` / `tfaccSync` 执行。更新命令和辅助镜像时包含 CPU cache 维护。

重定位表明确记录：镜像索引、字节/bit 位置、目标 region、偏移、引用跨度、对齐、地址高/低位及移位。每次从不可变模板创建新结果，重绑不会累积旧地址。它检查 40-bit bus、完整操作数跨度及同一个 4 GiB window；命令 buffer 自己的 driver 分配单独处理。LUT 内的地址也必须登记，不能只改命令中的 A/C 字段。

实例的 command/LUT 独立分配，权重通过调用方共享；禁止在设备执行期间 Rebind 或销毁。调用方须先轮询所有 launch core 完成，再对每个 core 调用 `tfaccSync(core, 1)` 完成 NPU cache clean/invalidate，之后才可以重写或释放这些 buffer。Rebind 内的 CPU cache 维护不能代替这一步。一个实例的运行及重绑由调用方串行化，输入/权重/manager 必须活到完成。当前 ABI 标记为 `NPU40T/BlasOp32/TFLLM-v1`，仍要求编译器/后端来自匹配版本；它不是任意 SDK 二进制间的兼容承诺。

**目前各原生结构的 NPU 路径使用 JIT 编译缓存。** `PackageWriter::AddProgram` 和 `PreparedDescriptor` 已支持保存、加载、重绑外部提供的显式命令计划；**从现有 MatMulHelper 自动导出完整重定位表，以及按 Transformer 层自动选择离线命令包，还没有接入**。不从不透明 BlasOp 字节猜测哪些数值是物理地址，也不将 JIT 描述符裸 dump 伪装成可迁移模型。

## 回归

2026-09-11 本地 macOS arm64 验证：6 项 CTest 全部通过；核心、FP16、调度及注入 SDK 的 lease 回归通过 ASan/UBSan，FP16、调度及 lease 回归也通过 ThreadSanitizer。六种结构的 F32 小 GGUF 与官方 llama.cpp 在 prefill/decode/prefill、滑窗历史、前缀回退场景对照，最大 logits 绝对误差 `5.33462e-4`。这些是确定性小模型结构回归，未下载或执行真实 14B/27B 权重。

- 核心：FP16、GQA/RoPE/causal 分块等价、KV stride/前缀/故障回滚、模型包及重定位；新增 F32/F16/UINT8 的 K/N 分块、非对齐尾部、重复输入更新与 FP32 结果对照。
- 结构：Llama 的 RoPE 因子/linear scaling/FFN bias，Qwen2 QKV bias，Qwen3 QK norm，Gemma embedding/GELU/tied head，Gemma2/3 post-norm/SWA/softcap；各结构 tiled attention 与 scalar 对照。
- GGUF：六种结构原生转换、UINT8 包读回；Qwen2 的 F16/Q4_0/Q8_0 解码后权重逐元素核对；Phi3 和 YaRN 明确回退；分片导入/执行/总参数量/跨分片身份校验；仅用超大 GGUF 头验证 >30B 在权重分配前拒绝。
- 量化 GGUF 接力：CPU ggml 量化 kernel 同时量化激活，TFLLM reference 使用 FP32 激活，测试容差为 0.02，观察到最大 logits 差 `0.00807317`；不能等同于浮点路径误差或真实模型精度。
- 调度：模拟 1..4 对 NPU，验证四组重叠执行、先空闲组领取新任务、FIFO/满队列、工作线程完整生命周期、初始化失败、故障组退休及最后一组故障排空；8 个会话/两份模型交错请求与串行 logits/KV 对照，析构完成已接收请求。真实 llama 回归还覆盖四个独立 context 并发 prefill/decode/prefill，NPU 部分由同接口的 FP32 Linear 模拟。新增 `tfllm-llama-batch-test_profile` 验证共享 context 的四路 FP16/Q4_0/Q8_0 decode 合批、不同历史长度、稀疏槽号、短 prefill 混合批、加入/退出、native prefill 重叠执行、失败回滚及日志批次宽度；架构与图像回归包含多序列 SWA/MRoPE。板端吞吐及大模型精度仍需实测。
- FP16：NEON 和强制标量两种构建均通过；覆盖 SIMD 尾部、subnormal/非有限量化输入、舍入、原地 RMSNorm、SiLU/GELU、masked Softmax、两种 RoPE、跨 K 大部分和抵消、FP16 类型贯通队列、KV 两种 stride、分块及并发。六种结构新增 FP16 prefill/llama decode/prefill/回退前缀对照；F32 源小模型最大 logits 差 `0.00120049`，Q4_0 源为 `0.00819582`，均为 CPU dot 模拟 NPU 的结构测试。
- KV 换入与重绑定：注入 SDK 检查快照准备不分配硬件内存、换入不量化、驻留命中、尾块、跨 K 累加、并发 head/会话隔离、同窗口跨层复用及不同窗口隔离、失败最多重试一次；真实 MatMulHelper 的主机命令回归检查大小核、batch、K=128/4160/12288 反复重绑定后命令和 LUT 完全一致，且不新增物理分配。这些检查不替代板端 cache/数值验证。`tfllm-npu-pool-test` 已加入静态层切换、原始动态与预处理 K/V 的混合并发运行。
- 锁生命周期：生产 NPU adapter 链接注入的 SDK，验证八核默认启用但空闲不持锁、不持锁编译/分配、外部 BUSY/超时恢复、每次执行释放、重复取得后的数值/计划复用、清理失败保留锁与内存。此测试不模拟实际 NPU cache 或驱动进程锁。
- 权重视图：注入 SDK 覆盖同一份 B 在 M=16/48/64/1024、不均匀 Grid 和并发请求间共享，检查新 M 不新增权重、A/B/C 同高位地址、分配失败诊断和重试。真实 MatMulHelper 命令回归核对大小核、192 列尾块、K=128/4160/12288 的 B stride/偏移/分段推进、反复重绑定和分配边界；板端 `tfllm-npu-pool-test` 增加变 M 的共享 B 数值与占用检查。这些主机检查不替代板端运行。
- 板端：NPU adapter、两种 NPU 测试、多会话示例及 SDK 的 `Excutor.cpp`、`PairLease.cpp`、`DeviceRuntime.cpp` 通过本地目标文件编译。整份 `NPU40T.cpp` 的 macOS 编译仍被原有 `InnerProductHelper.h:124` 的 float/double `std::max` 类型冲突阻挡，未宣称 SDK 在本机完整构建成功。`tfllm-npu-test` 包括 K=16448、N=16448、K/N 非对齐尾部；新增 `tfllm-npu-pool-test` 和 `tfllm-run-concurrent` 仍需板上实际运行，包括真实模型的精度、内存、性能验收。

重新生成 schema（需相同 FlatBuffers 版本）：

```sh
flatc --cpp --scoped-enums --gen-object-api -o TFLLM/schema TFLLM/schema/model.fbs
```

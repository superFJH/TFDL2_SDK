# PyTorch 模块化推理与自动接管

`tfllm.torch` 使用普通 CPU FP16 `torch.Tensor`，共享一个原生 NPU runtime。
不是新的 Torch device backend，不要求模型转 GGUF，也不依赖 TFDL graph/TFExecutor。
Python 自定义算子经小型 C ABI 借用 Tensor 指针；原生库不链接 LibTorch，调用时释放 GIL。

## 构建与使用

主机数值测试（明确选择 CPU，不会冒充 NPU）：

```sh
cmake -S TFLLM -B build/tfllm-torch -DTFLLM_WITH_TORCH=ON -DTFLLM_WITH_NPU40T=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/tfllm-torch -j
export PYTHONPATH="$PWD/TFLLM/python"
export TFLLM_TORCH_LIBRARY="$PWD/build/tfllm-torch/libtfllm-torch.so"
# macOS 使用 libtfllm-torch.dylib；需要 torch>=2.4，加载权重另需 safetensors
python TFLLM/tests/torch_modules_test.py
```

将 `TFLLM_TORCH_TEST_PYTHON` 设置为已安装 torch/safetensors 的 Python 路径，可以把主机模块测试
和真实 Python → 生产 NPU runtime → 注入 SDK 的端到端测试一起加入 CTest；测试假后端不会安装进 SDK。

板端可在 SDK 根构建启用 `WITH_TFLLM=ON`、`TFLLM_WITH_TORCH=ON`，自动链接本仓库 NPU40T target；
或独立构建设置 `TFLLM_WITH_NPU40T=ON`、`TFLLM_NPU40T_LIBRARY`、`TFLLM_NPU40T_CONFIG_INCLUDE`。
必须重建本仓库的 libNPU40T：原生卷积新增外部常驻 packed weight 绑定接口。
安装路径为 `tfllm/lib` 和 `tfllm/share/tfllm/python`，可将后者加入 PYTHONPATH。
也可在 `make install` 后的 `SDK/tfllm/` 中执行 `python -m pip install .`，
将 `tfllm.torch` 和原生库一起安装到 Python 环境，无需再设置上述路径。
`python -m pip install '.[torch]'` 额外安装 Torch/safetensors 依赖；完整说明见
[安装树 pip 打包](PIP.md)。
融合 attention 使用 **Torch C ABI 2**；原生 `libtfllm-torch` 和 Python 包需同步重建/安装，
新版 Python 会拒绝旧 ABI，避免把新增参数传入旧接口。
自动 attention 的 interval 扩展保持 ABI 2，旧 ABI 2 库缺少该符号时会要求重建，不会静默降级。
块掩码扩展同样保持 ABI 2，新增 `tfllm_torch_attention_blocks`；需要同步安装新版原生库和 Python 包。

## 自动接管现有模型

推荐先使用模型级 `optimize()`；无需修改受支持模型的 forward 或 `generate()`：

```python
import torch
from transformers import AutoModelForCausalLM
from tfllm.torch import NpuRuntime, optimize, optimization_report, restore

# 权重在 inference_mode 外加载；safetensors 仍由 Transformers 读取。
model = AutoModelForCausalLM.from_pretrained(
    "/models/my-qwen3", local_files_only=True,
    dtype=torch.float16, attn_implementation="sdpa",
).cpu().eval()

with NpuRuntime(chip=0) as runtime:
    optimize(model, runtime=runtime, strict=True)
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=32)  # inputs 是 CPU tensor
    print(optimization_report(model))
    report = optimization_report(model).to_dict()
    restore(model)
```

`optimize` 原地绑定实例 forward，**不替换原 Module/Parameter**：保留模块类型、hooks、
`state_dict` 键、共享模块及 tied weight。相同 Parameter 的同类算子共享 packed snapshot；
准备阶段预量化权重，版本变化或 `load_state_dict`/Parameter 替换后首次运行重新打包。
禁止 `.data` 绕过版本检测、并发改权重、并发 optimize/restore。暂时保留原权重及打包副本，
不承诺 CPU 内存只有一份权重。模型必须先 `.cpu().half().eval()`；训练、迁移设备、
deepcopy/整模型 pickle 或再次适配前应 `restore(model)`；保存/重载 `state_dict` 可直接使用。
不要在 `inference_mode` 内创建权重，否则没有版本计数器，无法安全检测修改。

当前接管范围：

| 入口 | 支持范围 | 保留在 PyTorch 的部分 |
| --- | --- | --- |
| 普通 `nn.Linear` / `nn.Conv2d` | 与下文原生算子限制相同；Conv 权重预打包 | bias、必要布局转换、其他非目标算子 |
| Transformers | 未改写的 Llama / Qwen2 / Qwen3 attention；prefill、GQA、DynamicCache decode、连续左右 padding | RoPE、KV cache 管理、归一化、采样等 |
| Diffusers | `Attention` + 原版 `AttnProcessor2_0`，self/cross attention | 原 processor 的 Q/K norm、投影、布局、residual；其中投影另被自动接管 |

Transformers 同时注册 attention 和 mask 接口，模型配置使用独立浅拷贝，不修改其他模型的配置。
常规 causal/padding mask 直接生成 `[B,L,2]` 可见区间；使用真实 query/KV offset，
增量多 token decode 不会错误套用 upper-left causal，也不先构建完整 `[B,H,L,S]` mask。
空可见区间输出零。Diffusers 仅在该 processor 调用内通过 `TorchFunctionMode` 路由 SDPA，
不全局替换 `torch.nn.functional`，不重写 processor 的其他模型语义。

外部 bool 或严格 `0/-inf` mask 只在每行可见 key 连续且各 head 相同的情况下融合；
任意有限 additive bias、padding 空洞、不同 head 的 mask、dropout、需要 attention weights、
packed/sliding/custom Transformers mask 不在首版自动融合范围。Wan/FlashVSR 的专用 processor、
LCSA、Conv3d、LoRA/offload 改写 forward、量化层、自定义 Linear 子类也不会被盲目替换。
特别注意：本适配不等于 FlashVSR 或任意 Diffusers pipeline 已经整模型兼容。

`strict=True` 对**识别到且要求接管的范围**拒绝不支持的静态配置/运行时语义；
`strict=False` 保留不支持的层，运行时回退会警告并记录。驱动/分配/数值错误始终向上抛出，
不会吞掉错误后在 CPU 重跑。报告区分静态 `routed/npu/cpu_reference/skipped/excluded` 和
实际调用 `npu/npu_fused/cpu_reference/fallback/rejected`；`routed` 不代表已经执行 NPU。
严格模式不是任意 Python/functional/custom op 的全覆盖证明；未知 attention 的检测也不是完整图审计。

可用 `operators=("linear", "conv2d")` 显式只接管投影/卷积，或用
`predicate=lambda name, module: ...` 排除指定模块；Transformers attention 必须整组选择，
不能因共享 mask 配置而仅适配其中一层。Diffusers pipeline 应对其中的 `transformer`/`unet`
分别调用，而不是把 pipeline 对象传给 `optimize()`。

## 函数式代码与 torch.compile

```python
from tfllm.torch import make_backend

backend = make_backend(runtime=runtime, strict=True)
compiled = torch.compile(model, backend=backend, fullgraph=True)
with torch.inference_mode():
    output = compiled(x)
print(backend.report)
backend.close()  # 释放 packed snapshot；之后执行可重新打包
```

这个初版后端支持捕获图里的 `F.linear`、`F.conv2d`、`F.scaled_dot_product_attention`，
以及完整的 `(Q @ K.transpose(-2,-1) * 正常量).softmax(-1) @ V`（也支持除常量或无 scale）。
必须没有中间 score/probability 外部消费者；匹配成功就删除整段中间矩阵，使用同一原生融合入口。
不猜测带 mask、cast、dropout 或其他变体的手写 attention；建议这些代码直接使用 SDPA。
未匹配的 MatMul/mm/bmm 在严格模式拒绝，非严格模式留给 Torch 并列为 skipped。
其他 CPU 算子原样执行；不是 Inductor 全算子编译器，也不是 `privateuse1` device backend。
原权重是图的运行时参数，第一次执行才打包，版本变化自动更新，不捕获示例 Tensor 的物理地址。

不要对同一模型叠加 `optimize()` 和此 compile 后端。建议 `fullgraph=True` 且不要启用
Dynamo `suppress_errors`；否则图断点/后端失败可能使框架执行图外原始代码，报告不能覆盖那些代码。
当前验证为 PyTorch **2.14.0**、Transformers **5.17.0**、Diffusers **0.40.0**；
核心手动模块的 torch>=2.4 要求不等于所有版本的框架自动适配都已验证。
框架依赖是可选的，只有使用对应适配时才导入。
扩展点参考：[HF attention/mask 接口](https://huggingface.co/docs/transformers/main/attention_interface)、
[PyTorch 自定义编译后端](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html)。

板端自动接管 sample（安装后 `tfllm/test/torch_auto.py`，无需下载模型）：

```sh
python TFLLM/examples/torch_auto.py --kind llama --tokens 1024 --repeat 5
python TFLLM/examples/torch_auto.py --kind diffusers --tokens 1972 --repeat 5
python TFLLM/examples/torch_auto.py --kind functional --tokens 1024 --repeat 5
# 或 --kind llama --model /models/my-qwen3 使用本地模型；--backend cpu 是主机参考
```

输出 `AUTO_CONFIG/AUTO_SETUP/AUTO_RUN/AUTO_RESULT/AUTO_COVERAGE`，包含静态/动态覆盖、
初始化耗时、热运行延迟及输出误差。compile 首次编译/打包是惰性的，至少保留一次 warmup；
sample 使用随机输入，误差不能替代真实任务质量验证。CPU reference/注入 SDK 延迟不是板端性能。

## 手动模块用法

```python
import torch
from safetensors.torch import load_file
from tfllm.torch import NpuRuntime, NpuLinear, NpuConv2d, NpuAttention

# 先加载实际权重，再准备硬件资源；以下 layer 已由应用构建。
layer.load_state_dict(load_file("linear.safetensors", device="cpu"))
layer = layer.half().eval()
with NpuRuntime(chip=0) as runtime:  # 默认 NPU；没有硬件会明确报错
    accelerated = NpuLinear.from_linear(layer, runtime=runtime)
    with torch.inference_mode():
        y = accelerated(x.cpu().half())
```

`NpuConv2d.from_conv2d(conv, runtime=runtime)` 与 `NpuAttention(runtime=runtime)` 使用同一 runtime。
`replace_linear(model, runtime=runtime, predicate=...)` 显式替换普通 Linear 子模块；不拦截 F.linear、
SDPA、量化层、自定义子类或 CUDA 专属代码；带 hooks 的层明确拒绝。共享/绑定权重需应用单独适配。

## 算子边界

- Linear：输入 `[..., K]`，输出 `[..., N]`，FP16 输入输出；激活逐行、权重逐输出通道对称 U8（零点 128）。
  权重提前量化、转置并驻留；输入直写工作空间，反量化直写输出 Tensor；超过 1024 行自动分块。
  CPU 模式为浮点 Linear 参考，不模拟 NPU 对称量化误差。
- Attention：SDPA 计算核心，输入 Q `[B,Hq,L,D]`、K `[B,Hkv,S,D]`、V `[B,Hkv,S,Dv]`。
  默认 `implementation="auto"` 在 NPU 上为无 mask / upper-left causal 调用选择原生融合路径。
  QK INT32 → C++ SIMD softmax/量化 → AV → FP16 输出直写 Tensor，不生成 Torch score/probability tile，
  沿用 TFLLM prefill/vision 的四个核对、双缓冲、命令模板重定位和缓存交接。
  `enable_gqa=True` 支持 Hq 是 Hkv 整数倍，原生每个 KV head 只准备一次 K/V，不 repeat/expand。
  支持 self/cross、非对齐 token/channel 尾部、Dv 不同于 D；scale 必须可表示为正有限 float32，S/D/Dv ≤ 16384。
  任意 bool/additive mask、非正 scale 或超出融合尺寸时，auto 会警告并使用旧的分块 Torch softmax 路径；
  `implementation="fused"` 严格拒绝这些情况，不静默回退；`implementation="torch"` 显式选旧路径。
  CPU backend 的 auto 仍用旧浮点参考路径，不冒充融合 NPU。`query_tile` 只控制旧路径，融合 tile 由 runtime 预算决定。
  K/V 每次调用创建不可变快照，不自动缓存跨帧状态。不含 QKV/output projection、RoPE、dropout、FlashVSR LCSA。
- 1×1 Conv2d：通过空间采样、NCHW/NHWC 转换复用 Linear。支持零 padding 和 stride，不进行空间 im2col。
- 非 1×1 Conv2d：**直接使用 NPU 原生空间卷积**，不进行 CPU im2col。
  整个 NCHW 输入共用一次 min/max 定标（范围包含零）；权重按输出通道非对称 U8。
  `scale=(max-min)/255`、`zero=clamp(round(-min/scale),0,255)`，全零范围用 scale=1。
  四舍五入采用 half-away-from-zero；真实/通道 padding 使用对应零点。
  原生输出已补偿零点的 INT32，CPU 乘输入与通道 scale 写 FP16；bias 在 Torch FP16 输出边界后相加。
  CPU 模式有独立的相同量化整数卷积 oracle。

第一版空间卷积限制：Conv2d/NCHW/OIHW、groups=1、dilation=1、两方向相同 stride 1..7，
kernel 每边 1..7、padding 每边 0..7、Cin×Kh×Kw≤4608；输入宽度超过 1024 必须为 32 的倍数。
不支持的配置明确报错，不隐式回退。尚无 Conv3d、ConvTranspose、depthwise/group convolution。
仅支持 CPU FP16 激活；BF16/FP32、非连续输入的 dtype 转换必须由调用方明确处理，非连续布局会 contiguous。
无 backward/autograd；应在 `torch.inference_mode()` 下使用。注册了 FakeTensor 形状函数，
手动 NpuModule 的通用 compile/export 不作承诺；自动图适配请使用上面的专用 `make_backend`。

## 融合 Attention 用法

```python
with NpuRuntime(chip=0, attention_fuse_workers=16,
                attention_score_mib=4, attention_pipeline=True,
                attention_lease_tiles=4) as runtime:
    attention = NpuAttention(runtime=runtime, implementation="fused")
    with torch.inference_mode():
        y = attention(q, k, v)                       # ViT / DiT / cross-attention
        y = attention(q, k, v, is_causal=True)       # upper-left causal
        y = attention(q, k, v, enable_gqa=True)      # Hq % Hkv == 0
```

直接函数式调用也可用 `runtime.attention(q, k, v, is_causal=False, enable_gqa=False)`，始终严格融合。
单 token decode 若 K/V 包含整个历史，通常应不传 causal：upper-left causal 的单行只看第一个 key，
不会自动按 KV cache 长度右对齐。需要明确区间时可调用
`runtime.attention_intervals(q, k, v, intervals, enable_gqa=True)`：intervals 为 CPU int64 `[B,L,2]`，
每行 `[begin,end)`，各 head 共用，空区间输出零。该入口直接验证区间，不推断 cache 语义；
Transformers 自动适配才负责生成带 query/KV offset 的区间。
原有 `NpuAttention(runtime=runtime)` 调用不用改即可对支持的形状启用融合。
这不是全局 monkey-patch，不会自动替换应用里的 `F.scaled_dot_product_attention`。

### 不连续块掩码（FlashVSR 等专用 attention 的计算核心）

新增显式 `NpuBlockSparseAttention`，支持不同 query head 各自选择多个不连续 KV 块。
每个 query 块在 C++ 收集所选 K/V，再调用一次原生融合 QK → softmax → AV；
**softmax 在该 query 块选中的全部 key 上统一归一化**，不是独立算各段再平均。
不生成完整 token 级 mask、QK 或 probability，不走 Python 分块 / Torch softmax。
沿用现有八核调度、核对流水、score tile 预算和命令模板重定位，没有修改 NPU 驱动。

```python
from tfllm.torch import BlockSparseMask, NpuBlockSparseAttention

# q/k/v 是 CPU FP16 [B,H,L,D]；先由模型完成 RoPE、窗口重排、KV cache 拼接。
# final_blocks 是模型已有策略生成的 CPU bool [B,Hq,Qblocks,KVblocks]。
# True 表示可见；传的是最终局部窗与 top-k 的交集，不是 token 级 [L,S] mask。
mask = BlockSparseMask.from_dense(final_blocks, query_block_size=128, kv_block_size=128)
attention = NpuBlockSparseAttention(runtime=runtime)
with torch.inference_mode():
    y = attention(q, k, v, mask, enable_gqa=True)
```

也可以直接构造 `BlockSparseMask(row_offsets, column_indices, 128, 128)`，省去块级 bool 转换，
或调用 `runtime.attention_blocks(q,k,v,row_offsets,column_indices, query_block_size=128,kv_block_size=128)`。
CSR 约定如下：

- offsets/indices 均为一维 CPU int64；行顺序为 `[B,Hq,ceil(L/query_block_size)]` 展平，
  offsets 长度为行数加一，首项 0、末项 nnz，单调不减。
- 每行 indices 是 KV 块编号，必须升序且不重复；同一 query 块所有 query 共享这些 key。
  不广播 batch/head；GQA 只负责映射 KV head，不合并各 query head 的掩码。
- 空行输出零；query/KV 尾块自动裁剪，D/Dv 尾部内部补齐。
  query 块大小 1..1024、KV 块大小 1..16384、D/Dv ≤16384，通常使用 128×128 块。
- **全局 S 可以超过 16384，但每个 CSR 行选中的真实 key 总数最多 16384**，超过会明确报错。
  当前未实现跨此上限的 online-softmax 合并。scale 必须为正有限 float32。
- 掩码已经表达最终可见性；没有额外 causal、additive bias、dropout 或 backward。
  块内若还需要 token 级不同可见性，不能直接用一个粗粒度块表示。
- CSR 每次调用验证并快照，Q/K/V 借用到原生任务全部结束；调用期间不要修改这些 Tensor。
  非连续输入会 contiguous；不同调用之间可更新 mask/KV，不复用旧数据。NPU 不支持时明确报错，不回退 CPU。

此入口**不是自动适配 FlashVSR 整模型**：窗口划分、2D 局部窗、top-k 策略和时序 cache 仍由应用负责，
将原来的稀疏 attention 核心换成上述调用即可；其他算子可另用
`optimize(model, runtime=runtime, operators=("linear", "conv2d"))`。
`optimize()` / `make_backend()` 尚不会把任意散射 token mask 自动改写为 CSR。

首版会对每个 query 块收集并重新量化所选 KV，暂未复用不同 query 块重叠的 packed KV；
稀疏度不高或块很小时，gather/prepare 和提交开销可能抵消收益，需要板端测量。
动态 V 的量化范围来自当前选中集合，可能不同于全局 dense attention，不能要求两条 INT8 路径逐位相同；
应对照 FP32 和真实视频质量。profile 构建新增 `torch.attn_blocks_fused`、
`torch_sparse_kv_gather`、`torch_sparse_kv_prepare` 范围。

板端 sample（安装后 `tfllm/test/torch_block_attention.py`）：

```sh
python TFLLM/examples/torch_block_attention.py --tokens 1024 --keys 32768 \
  --heads 4 --kv-heads 2 --head-dim 64 --selected-blocks 8 --repeat 5
# 尾部、空行和不同 Dv：
python TFLLM/examples/torch_block_attention.py --tokens 257 --keys 32781 \
  --head-dim 72 --value-dim 80 --empty-every 7 --repeat 5
```

输出 `BLOCK_CONFIG/BLOCK_LOGICAL_QK/BLOCK_RUN/BLOCK_SUMMARY`，记录含 KV 准备的同步延迟和
抽样 query 行的 FP32 误差，参考计算不分配完整 L×S 矩阵。逻辑 QK MiB 不是实测内存峰值/流量。
sample 掩码为合成的各 head 独立散射块，不代表 FlashVSR 的真实选块算法；请再用真实模型的 mask/QKV 验证。

### 调优与常规融合测试

`attention_fuse_workers` 为 1..32（默认 16，包含核对调用线程），共享 CPU 线程池只增不减；
线程数 A/B 应分别启动进程。`attention_score_mib` 为 1..16（默认 4），是一个 slot 跨四个
pair lane 的 INT32 score 预算，双缓冲使用两个 slot，最小 16 行对齐可能向上取整。
它不是整个 runtime 的内存上限；还有 K/V、UINT8 激活、AV 输出和命令缓冲。
Torch 与原生 vision 共用最多两个并行 head job，单次调用最多提交两个，异常时全部等待结束。
`attention_lease_tiles` 为 1..16（默认 4），控制一个核对连续处理多少 tile 后归还 lease。
小形状可能只启用一部分核对或单缓冲，不保证八核始终满载。

板端测试（源码 sample；安装后在 `tfllm/test/torch_attention.py`）：

```sh
python TFLLM/examples/torch_attention.py --tokens 1972 --heads 16 --head-dim 72 \
    --warmup 3 --repeat 10 --fuse-workers 16
python TFLLM/examples/torch_attention.py --tokens 513 --keys 1024 --heads 8 --kv-heads 2 \
    --head-dim 72 --causal --score-mib 1
```

sample 交替运行旧 Torch softmax 路径与融合路径，输出延迟、加速比、旧路径误差及 FP32 抽样误差。
不构造完整 FP32 L×S 参考矩阵。新路径保持 TFLLM 的 FP16 score/probability 舍入边界，
与旧 Torch FP32 softmax 不要求逐 bit 一致；真实模型仍应验证任务精度。注入 SDK 的时间不能作为板端性能。

## 八 TFACC 调度和生命周期

一个 runtime 预留一份物理内存池，使用 TFLLM 原有四个核对线程管理八个 TFACC。
GEMM 由已有 Grid 策略按 M/N 切分；原生 Conv 按输出通道与高度分片，包含正确输入 halo。
每个分片准备 16/8 MAU 变体，任务在核对之间调度。能否同时用满八核取决于形状和竞争，
不会为不足以有效拆分的小算子伪造八路工作。多个 module 不能各创建一套 runtime。

静态权重与在途工作空间分离；并发执行命令和 A/C 缓冲私有，完整等待后回收。
错误/超时沿用原有芯片隔离，不释放可能仍被 DMA 使用的资源。close() 等待本 runtime 的活动调用结束。
state_dict 保存 CPU 权重，不保存物理地址；加载后显式 prepare() 或首次 forward 重建 packing。
普通原地权重修改通过 PyTorch version counter 检测；禁止 `.data` 绕过版本机制及并发修改/推理。
模块 close() 释放其 packed handle，下一次 forward 可重建；runtime close() 则永久关闭全部模块。

当前保留 CPU 权重副本以支持 checkpoint，不能把常驻占用简单等同于 INT8 权重大小。
原生 Conv 权重只准备一次；命令采用与 MatMul 相同的 **CPU 不可变模板 + 每次执行私有缓冲重定位**。
首次遇到卷积几何形状时编译 8/16 MAU 两套配方，缓存按形状而非权重、输入 zero-point 或物理 HIGH 区分。
热路径只复制命令镜像、重定位输入/输出/权重/LUT/临时缓冲地址，并更新 padding 命令和 bias 记录中的 zero-point；不重新编译。
工作区及命令空间按配方的实际需求申请，执行结束回收，不保留空闲物理工作区。新形状或 CPU 模板被 LRU 淘汰后仍会冷编译。
原有原生卷积形状限制不变；例如部分超宽 stride-2 形状仍受 SDK 输出地址对齐限制。
没有板端吞吐、延迟和模型质量实测前不承诺性能。

## 验证

`torch_optimize_test.py` 分别在 CPU reference 和生产 NPU runtime + 注入 SDK 上验证模块身份、
绑定权重、hooks、重载/版本刷新、回滚、strict/fallback、interval/空行/GQA、函数式 compile 与
整段 attention 匹配；安装可选依赖后还运行真实 Llama/Qwen2/Qwen3 的 prefill/cache/generate、
attention weights 回退，以及 Diffusers self/cross/QK norm。缺失可选依赖时对应 case 明确 skip。
C++ dense attention 对非零 begin、batch 不同区间和空行另做逐 bit 参考、并发及无残留 workspace 检查。

`torch_modules_test.py` 在真实 PyTorch 上测试 Linear、多批/非连续输入、safetensors 重载、
两条 Conv 路径、非对称量化参考、cross/causal/masked Attention、动态 KV、并发、错误边界和 FakeTensor。
`torch_npu_test.py` 在真实 Torch → C ABI → 生产融合 runtime → 注入 SDK 上验证融合 self/cross/causal/GQA、
非连续输入、不同 D/Dv、双缓冲尾块、动态 K/V、零值、并发、严格拒绝边界及关闭后调用；
测试禁止 Python pack/linear/softmax，确保没有走旧路径。C++ head 测试另验证融合结果与原生
FP16 分阶段路径逐 bit 相等、模板复用、申请失败回滚和零空闲工作空间。
`tfllm-npu-lease-test{,_profile}` 用注入 SDK 执行生产 NPU runtime，验证矩阵/原生 Conv 分片、
逐次零点变化、INT32 数值、Conv 热调用零编译、同形状跨权重/HIGH 并发、冷编译去重、
命令容量/绑定失败回滚、常驻权重释放、零空闲 workspace、核对 lease 及失败隔离；不模拟真实硬件缓存。
`tfllm-npu-command-test` 使用真实 SDK 编译器，将重定位结果与独立重编译结果逐字节对比，
覆盖 8/16 MAU、padding zero-point、输入偏移、HIGH、LUT、原生通道/宽度分块及临时 bias，验证 CPU 模板/实例化不申请 DMA 内存。
板端仍需对比 PyTorch 整模型误差，并分别测量 packing/量化/命令准备/NPU/dequant/CPU 算子的耗时。

本次主机验证使用 macOS arm64：独立构建及 CTest、ASan/UBSan、安装后自动找库均通过。
SDK 内 NPU40T + tfllm-npu40t + tfllm-torch 的编译链接也通过（GCC 15，复用已有 build/Release 生成头文件）。
这不等于板端硬件执行验证；本机没有 TFACC。原有 SDK 全量目标在 AppleClang 下还存在
InnerProductHelper 的 float/double 模板匹配问题，与本适配层无关，本次未修改。

# PyTorch 模块化推理与自动接管

`tfllm.torch` 使用普通 CPU FP16/FP32 `torch.Tensor`，共享一个原生 NPU runtime。
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

SAM 与 Depth Anything 的任务入口见 [VISION.md](VISION.md)。`optimize()` 默认
`precision="auto"`，按外部 CPU FP16/FP32 输入、权重和输出类型选择通道。
FP32 原值直接在线量化，I32 直接反量化到 FP32，不经过 FP16 舍入或完整 FP16 激活。
该规则也适用于手动模块、runtime 和 `make_backend()`；不迁移模型设备或修改其它 Torch 算子。

输出跟随输入 dtype；自动接管时尊重 CPU autocast 的 FP16/FP32 输出要求。Q/K/V 必须同 dtype。
本轮没有新增 BF16 通道；auto 拒绝 BF16。`precision="strict"` 显式要求 FP16，
旧 `precision="fp16"` 名称继续兼容 FP16/FP32 及原有 BF16→FP16 桥接，FP32 不再因这个名称缩窄。
`strict=True/False` 控制算子覆盖，和 `precision="strict"` 是两个独立选项。

这些 dtype 指 Torch 边界及 CPU 量化/反量化，NPU 仍执行 W8A8/I32。
TFLLM 原生 LLM serving 的 Session、prefill/decode、KV 和内部 FP16 数值边界继续使用原接口；
新增外部 FP32 接口只由 Torch 路径选择。原生库与 Python 包需同步更新，旧 ABI2 库缺少
`tfllm_torch_external_dtypes` 能力时，FP32 调用明确要求重建，不静默退回 FP16。

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
不承诺 CPU 内存只有一份权重。模型先 `.cpu().eval()`，保留所需 FP16/FP32；训练、迁移设备、
deepcopy/整模型 pickle 或再次适配前应 `restore(model)`；保存/重载 `state_dict` 可直接使用。
不要在 `inference_mode` 内创建权重，否则没有版本计数器，无法安全检测修改。

当前接管范围：

| 入口 | 支持范围 | 保留在 PyTorch 的部分 |
| --- | --- | --- |
| 普通 `nn.Linear` / `nn.Conv2d` / `nn.ConvTranspose2d` | 与下文原生算子限制相同；Conv 权重预打包 | bias、必要布局转换、其他非目标算子 |
| 普通 `nn.GELU` / `nn.SiLU` | CPU 快速激活；默认按 dtype/元素数选择原生核或 Torch | 小张量和板上更快的 FP32 Torch 核 |
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

## 快速逐点激活

`optimize()` 默认目标包括 `gelu/silu/sigmoid/tanh/mish/softplus/elu/selu`，对应普通 `nn.Module`。
保持模型类型、hooks、共享模块和 `state_dict`；`restore()` 可恢复原 forward。
`make_backend()` 接管对应函数式调用及已识别的非原地 ATen 调用，也支持 Tensor 的 `sigmoid()/tanh()`。
自定义激活模块需通过编译捕获其函数式调用，不能按类名猜测。

```python
optimize(model, runtime=runtime, operators=("gelu", "silu"))
with torch.inference_mode():
    y = model(x)
    a = runtime.gelu(x)                       # 显式使用原生 CPU 核
    b = runtime.gelu(x, approximate="tanh")
    c = runtime.silu(x)                       # 支持 inplace=True
    d = runtime.mish(x)
    e = runtime.softplus(x)                   # beta=1, threshold=20
```

`activation_policy="auto"` 是 `optimize()` / `make_backend()` 的默认值：依据 65 上的 A77
实测，按下面的元素数门槛选原生核，其余保留 Torch：

| 算子 | FP16 | FP32 |
| --- | ---: | ---: |
| erf GELU | 4096 | 保留 Torch |
| tanh GELU / SiLU / Sigmoid | 65536 | 保留 Torch |
| Tanh | 65536 | 262144 |
| Mish | 4096 | 4096 |
| Softplus / ELU / SELU | 65536 | 65536 |

可用 `activation_policy="native"` 或 `runtime` 的同名方法强制测试原生核，`"torch"` 强制保留 Torch。
这些是保守的固定门槛，其他 CPU/Torch/线程配置需重新测量。报告的实际调用分别标为
`cpu_fast` 和 `cpu_torch`，激活本身不占用 NPU。

数学核源自 TFDL `Fp16EvalUtils.h` 的稳定 sigmoid、erf 和 NEON exp；FP16 默认进一步使用
65536 项精确编码索引，每算子占 128 KiB，一次生成后共享，只存 FP16 结果。没有输入量化，
不生成整张 FP32 临时 Tensor。表项按 FP32 公式计算并舍入一次，并非承诺与各版本 Torch
逐位相等。`TFLLM_TORCH_ACTIVATION_LUT=0` 在进程启动前设置，可对照 NEON 算术路径。
FP32 直接执行 NEON 向量公式，无 FP16 缩窄；默认 erf GELU 使用 TFDL 的 A&S erf 近似，
与 tanh GELU 分开，板测常规输入的最大绝对误差低于 `1e-6`。
Mish 的 FP32 核使用等价的单 exp 有理式，消除 log 和第二次 exp；Softplus 用 `exp(-abs(x))`
与稳定的 log1p 级数，保留负数尾部。各 FP16 表按需独立初始化，不会在创建 runtime 时生成全部表。

GELU 保留 `approximate="none"/"tanh"`；SiLU/Mish/ELU/SELU 的原地接口返回输入，保留 view 和 Torch
版本计数器检查。激活都跟随输入 dtype，CPU autocast 不会把 FP32 激活强制转半精度。
原生 Softplus 支持 `beta=1, threshold=20`，ELU 支持 `alpha=1`；auto 会将其他有效参数保留在
Torch，并读取模块属性的当前值。显式 native 策略/原生 runtime 方法会拒绝不支持的参数，避免错误计算。
仅用于推理；自动接管在训练/梯度模式下按 strict 设置拒绝或回退原实现。
原生 ABI 仍为 2，新增 `tfllm_torch_activation_f16/f32`，缺少符号时明确要求同步重建安装。
新增六种激活还要求 `tfllm_torch_activation_version() >= 2`；旧 GELU/SiLU ABI2 库不会被误当作支持新 ID。

可复现的公共 API 精度/速度对照（计入分配、Python/C ABI，预热后交错 A/B）：

```sh
PYTHONPATH=TFLLM/python TFLLM_TORCH_LIBRARY=/path/libtfllm-torch.so \
  python TFLLM/examples/torch_activation_benchmark.py --threads 8 --output activations.json
```

65 上的原始数据和测试记录见 [GELU/SiLU 测量报告](artifacts/torch-activation-20261017-server65/REPORT.md)
和[其他激活测量报告](artifacts/torch-activation-extra-20261017-server65/REPORT.md)。
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

### 固定模型的 CPU 编译融合与请求并发

已 `optimize()` 的纯 eval 模型可额外建立 `compile_snapshot()`：
它捕获 TFLLM 原生算子调用，再用 Inductor 融合 CPU 部分，保留显式输入/权重 dtype。
该可选接口已在 Torch 2.14 上验证，普通 eager 导入不依赖编译器模块。

```python
from concurrent.futures import ThreadPoolExecutor
import torch
from tfllm.torch import NpuRuntime, optimize, restore, compile_snapshot

torch.set_num_threads(16)  # 在主线程设置一次，所有请求共享这项配置
model.eval()
with NpuRuntime(chip=0, spatial_convolution="native", memory_fraction=.125) as runtime:
    optimize(model, runtime=runtime, operators=("conv2d", "linear"), precision="auto")
    fast = None
    try:
        with torch.inference_mode():
            fast = compile_snapshot(model, example_input)
        def infer(value):
            with torch.inference_mode():
                return fast(value)
        with ThreadPoolExecutor(max_workers=4) as requests:
            outputs = list(requests.map(infer, inputs))
        print(fast.report)  # 包含准备、捕获及首次编译耗时
    finally:
        if fast is not None:
            fast.close()
        restore(model)
```

每个请求线程一次只执行一个样本。只读 eval 模型和 runtime 共享，原生执行工作区由
runtime 独占借出，输入、输出和观察统计保持私有。关闭快照会等待该快照的在途请求完成，
不会关闭调用者拥有的 runtime。实例数需要实测；8 个请求不保证比 4 个更快。

`chip=0` 指第一个八核 NPU 芯片，调度范围为 TFACC 0–7，含四个 pair worker；
不是只使用第 0 核。`torch.set_num_threads(16)` 也不是整个进程的 OS 线程上限，
runtime 有独立的 CPU 辅助线程。单请求延迟与多请求吞吐应分别对照 ORT。

快照固定输入形状、stride、dtype/device 和模型 Python 配置。输入数据可变；输入相关的
`item()`/`tolist()` 或 Python 分支会被拒绝。权重 reload/原地修改、模块替换、训练模式、
`restore()` 后需重建快照；任意自定义 Python 状态仍应保持不变。
编译融合可能改变浮点舍入，再经过动态量化会放大局部差异，需用任务数据验收。
该接口不会把原生近似动态量化变成精确 FP32，也不会静默回退到 CPU 模型。

首次捕获/编译可能需要数十秒，`report` 单独记录；稳态测速应排除这项启动成本，
端到端校准时间应包含它。少量校准图片通常优先使用 eager 请求并发。

深层 CNN 可按显存预算增加执行实例缓存，避免默认 32 项的 LRU 在层间反复替换：

```sh
export TFLLM_NPU_WORKSPACE_CACHE_INSTANCES=512  # 1..4096，默认 32
export TFLLM_NPU_WORKSPACE_CACHE_MIB=2048       # 1..4096 MiB，默认 1024
```

计数上限独立于字节预算；空闲超时和内存压力回收继续生效。增加上限不会在启动时预分配
所有实例。此选项需更新配套原生库；C++ 客户端使用更新的 `NpuOptions` 头文件时需重新编译。

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
from tfllm.torch import NpuRuntime, NpuLinear, NpuConv2d, NpuConvTranspose2d, NpuAttention

# 先加载实际权重，再准备硬件资源；以下 layer 已由应用构建。
layer.load_state_dict(load_file("linear.safetensors", device="cpu"))
layer = layer.cpu().eval()  # FP16/FP32 均可，保持调用方精度
with NpuRuntime(chip=0) as runtime:  # 默认 NPU；没有硬件会明确报错
    accelerated = NpuLinear.from_linear(layer, runtime=runtime)
    with torch.inference_mode():
        y = accelerated(x.cpu())  # 输出跟随 FP16/FP32 输入
```

`NpuConv2d.from_conv2d(conv, runtime=runtime)`、`NpuConvTranspose2d.from_conv_transpose2d(deconv, runtime=runtime)`
与 `NpuAttention(runtime=runtime)` 使用同一 runtime。
`replace_linear(model, runtime=runtime, predicate=...)` 显式替换普通 Linear 子模块；不拦截 F.linear、
SDPA、量化层、自定义子类或 CUDA 专属代码；带 hooks 的层明确拒绝。共享/绑定权重需应用单独适配。

## 算子边界

- Linear：输入 `[..., K]`，输出 `[..., N]`，FP16/FP32 自动分派；激活逐行、权重逐输出通道对称 U8（零点 128）。
  FP32 输入/权重直接按原值量化，FP32 结果不经过 FP16 舍入；bias 转为输出 dtype 后相加。
  权重提前量化、转置并驻留；输入直写工作空间，反量化直写输出 Tensor；超过 1024 行自动分块。
  CPU 模式为浮点 Linear 参考，不模拟 NPU 对称量化误差。FP32 参考也遵守
  `TFLLM_CPU_THREADS`；直接读取 FP32 权重，其它格式每次调用只解码一次，AArch64 使用 FP32 NEON 归约。
  FP16 参考保留原有舍入边界；这些参考内核不等同于 Torch/BLAS 的优化 GEMM。
- Attention：SDPA 计算核心，输入 Q `[B,Hq,L,D]`、K `[B,Hkv,S,D]`、V `[B,Hkv,S,Dv]`。
  默认 `implementation="auto"` 在 NPU 上为无 mask / upper-left causal 调用选择原生融合路径。
  QK INT32 → C++ SIMD softmax/量化 → AV → 按输入 dtype 直写输出，不生成 Torch score/probability tile，
  FP32 的 Q/K/V、score、probability 与输出均不缩窄；FP16 通道保留既有舍入边界。
  沿用 TFLLM prefill/vision 的四个核对、双缓冲、命令模板重定位和缓存交接。
  `enable_gqa=True` 支持 Hq 是 Hkv 整数倍，原生每个 KV head 只准备一次 K/V，不 repeat/expand。
  支持 self/cross、非对齐 token/channel 尾部、Dv 不同于 D；scale 必须可表示为正有限 float32，S/D/Dv ≤ 16384。
  任意 bool/additive mask、非正 scale 或超出融合尺寸时，auto 会警告并使用旧的分块 Torch softmax 路径；
  `implementation="fused"` 严格拒绝这些情况，不静默回退；`implementation="torch"` 显式选旧路径。
  CPU backend 的 auto 仍用旧浮点参考路径，不冒充融合 NPU。`query_tile` 只控制旧路径，融合 tile 由 runtime 预算决定。
  K/V 每次调用创建不可变快照，不自动缓存跨帧状态。不含 QKV/output projection、RoPE、dropout、FlashVSR LCSA。
- 1×1 Conv2d：NPU 默认直接读写 NCHW，量化与原 Linear 桥接一致：输入按每个像素的
  `max(abs(x))/127` 对称定标，权重按输出通道对称定标，U8 zeroPoint=128；全零 scale=1。
  NPU 只使用原生 Conv 命令，Cin≤4096、Cout≤16384；不再回退到 GEMM。
  stride=1、padding=0 且满足原生空间限制时直接执行；其他 stride/padding、宽度尾部和
  大空间平面先在 C++ 中按最多 8192 像素的 NCHW 块取样补零，再执行原生 1×1。
  不生成完整 NHWC 或 padding 张量；每像素定标使空间分块不改变量化结果。
  输入 U8 与输出 I32 均为 NCHW，反量化乘像素 scale 与权重通道 scale。
  FP16 输出舍入和后续 FP16 bias 相加保持原来的边界。
  旧的 `TFLLM_NPU_POINTWISE_NCHW` 路由开关已移除；设备异常直接报错，不回放计算。
  FP32 输入现在直接按 FP32 原值求每像素 scale 并量化 U8，I32 直接反量化写 NCHW FP32，
  不创建整图 FP16 输入/输出，也不在寄存器内舍入到 FP16。FP16/FP32 权重按其原值逐输出通道量化。
  `NpuConv2d(x_fp32)` / `forward_fp32(x_fp32)` 与 `optimize(...)` 的 FP32 1×1
  均走此入口；FP32 bias 在输出后按 FP32 相加。这里的 FP32 激活语义与旧的 FP16 桥接不同，
  需用直接 FP32 量化参考验证；极小的非零 scale 下限为 `FLT_MIN`，全零仍用 1。
  显式 CPU backend 使用同 dtype 的 NCHW 浮点卷积参考；旧版库缺少入口时要求重新编译。
  板端复测可运行 `python TFLLM/examples/torch_pointwise_benchmark.py`；它交替计时新旧路径，
  在计时外检查 FP16 一致性，并报告直接 FP32 与原 FP32 桥接的配对耗时。
  `TFLLM_NPU_VERIFY_OUTPUTS=1` 用于单独的 I32 检查，不用于性能计时。
- 非 1×1 Conv2d 默认使用 NPU 原生空间卷积，不进行 CPU im2col。
  整个 NCHW 输入（含全部 batch）共用一次 min/max 定标（范围包含零）；权重按输出通道非对称 U8。
  大输入并行扫描后归约成同一个范围；低通道输入沿空间分块量化，避免 RGB 仅有三个 CPU 任务。
  FP16/FP32 保留各自原有的 scale 舍入和量化规则；这些优化不改变权重量化或输出数值。
  `scale=(max-min)/255`、`zero=clamp(round(-min/scale),0,255)`，全零范围用 scale=1。
  四舍五入采用 half-away-from-zero；真实/通道 padding 使用对应零点。
  原生输出已补偿零点的 INT32，CPU 乘输入与通道 scale，直接写入输入对应的 FP16/FP32 输出；
  FP32 输入范围及权重量化直接读取 FP32，bias 在相应输出 dtype 下相加。
  CPU 模式有独立的相同量化整数卷积 oracle。
- 可选 `NpuRuntime(spatial_convolution="gemm")`：空间 Conv 使用分块 im2col + NPU Linear，
  权重逐输出通道对称 U8、输入逐 patch 对称 U8，复用 Linear 的 INT32 执行路径。
  每个 batch 逐输出行段处理，通常至多 1024 个 patch（输出宽度大于 1024 时至少一整行），
  不在推理路径构造整图展开矩阵；padding 使用实数零，bias 在所选输出 dtype 下相加。
  原始权重不变，但量化规则与 native 路径不同；质量和实际速度需要重新验收。
  这个选项只更改 Python 桥接，使用已有的 native Linear ABI，不需要为该选项重编 C++ 库。

2026-10-14 的真实板端 probe 已发现空间 Conv 在 stride=2 和 stride=4 下均有 INT32
不匹配，stride=4 切换到旧同步缓存维护后仍出现同一错排。
已修复一个能精确解释 stride=4 dump 的命令生成缺陷：输入宽度 ≥1024 时按 512 列切分，
INT32 bypass 把块内多行连续写进完整输出，覆盖另一宽度块的列并留下未写位置。
现在匹配常规整宽布局的 stride 1/2/4 子卷积先写紧凑 scratch，保留原输入行距，
再按通道/行复制回完整输出。真实 SDK 的输出覆盖、地址和重绑定检查通过；
2026-10-14 修复后的板端重放已通过：输入 `[1,3,1024,1024]`、输出通道 96、
3×3/stride=4/pad=1，在 `TFLLM_NPU_VERIFY_OUTPUTS=1` 和 8 核运行配置下连续三次
与量化参考逐值一致（cos=1、relative_l2=0、max_abs_error=0），重复输出也逐 bit 一致。
首次调用 732.13 ms，后两次 7.31/7.11 ms，均包含输出验证；不能据此拆出纯硬件耗时。
该输入的量化参考相对 FP32 的 relative_l2=0.005990、cos=0.999982。
stride=2、7×7 及整模型仍需复测，尚不能确认所有硬件错误已消除。
此修复需要重编部署 `libNPU40T`；recipe build ID 已更新，旧命令缓存会失效。
可用下面的可选 GEMM 路径重放输入，作为独立对照：

```sh
TFLLM_NPU_VERIFY_OUTPUTS=1 python TFLLM/examples/torch_conv_probe.py \
  --replay /tmp/conv_failure.pt --spatial-impl gemm --repeat 3
```

更新 `tfllm.torch` Python 包及 probe 后运行，沿用现有 `TFLLM_TORCH_LIBRARY`。
probe 的 GEMM 参考单独使用逐 patch/逐输出通道对称量化，同时报告与 FP32 卷积的误差。

第一版空间卷积限制：Conv2d/NCHW/OIHW、groups=1、dilation=1、两方向相同 stride 1..7，
kernel 每边 1..7、padding 每边 0..7；3×3 的 Cin×Kh×Kw≤33025（保留完整 INT32 累加范围），其他 kernel 暂保留≤4608；native 路径输入宽度超过 1024 必须为 32 的倍数。
3×3 超过 4608 的归约由 SDK 沿输入通道拆分，各块在 NPU 上计算并以 INT32 累加，最后统一反量化；
例如 3×3、Cin=768 的 K=6912 可直接使用原生卷积。权重仍按完整输出通道逐通道量化，并复用命令模板。
空间分片的 halo 后输入起始偏移必须按 4 字节对齐（Conv 命令的低两位是控制位）。
切分器选择合法的输出行边界；stride/padding 无合法分界时减少高度分片，不移动像素、不额外改 padding。
native 路径仍使用缓存命令模板和每次执行重定位。172×114 等非 4 倍数宽度已加入回归。
不支持的配置明确报错，不隐式回退。尚无 Conv3d、depthwise/group convolution。
ConvTranspose2d 使用 SDK 原生 Deconvolution，输入/输出为 NCHW；state_dict 保留 PyTorch IOHW 权重，
打包时转换为 SDK 的 OIHW 并翻转空间 kernel。支持 stride=(2,2)、groups=1、dilation=1、
kernel 每边1..7、Cin×Kh×Kw≤4608、各方向 padding 小于对应 kernel、output_padding 每边0或1，
输入宽度≤512、去掉 output_padding 后的基础输出高宽必须为正。
FP16/FP32 权重和激活分别按原值仿射量化，U8→I32 后按输入 dtype 输出，bias 在输出域相加。
SDK 偶数 kernel 的奇数输出宽度存在行布局问题：NPU 计算基础矩形，新增 output_padding 右列/底行
使用同一组 U8 操作数补算精确 I32 累加；没有完整 im2col 或 FP16 桥接。
`output_size` 可选择合法的四组 stride-2 输出尺寸；打包缓存随权重版本刷新。

`optimize()` 默认也选择 `conv_transpose2d`；vision.attach 同样接管受支持层。
`make_backend()` 在捕获图中直接拒绝非 stride=(2,2) 的 conv_transpose2d，即使 strict=False 也不静默回退。
手动模块在构造时拒绝非 stride=2；其它不支持的静态配置按既有 strict 策略处理，运行时几何越界明确报错。
反卷积按输出通道等份分核，不切 H；小任务根据工作量减少核数。
复用私有命令和工作区，动态更新输入量化参数；遵守现有工作区预算和压力回收。
该 SDK 接口没有可重定位的 Deconvolution recipe，反卷积使用原生 SDK 缓存交接，暂不参与磁盘 recipe 缓存。

```python
from tfllm.torch import NpuConvTranspose2d
layer = NpuConvTranspose2d.from_conv_transpose2d(model.upsample, runtime=runtime)
with torch.inference_mode():
    y = layer(x)  # 输出 dtype 跟随 FP16/FP32 输入
```

固定随机数据的板端基准：`python TFLLM/examples/torch_deconv_benchmark.py`；
包含 SAM 常用的两级 kernel=2/stride=2 上采样，报告冷调用、预热耗时与 Torch FP32 误差，不下载模型。

手动模块支持 CPU FP16/FP32 激活和权重，按输入 dtype 返回结果；FP16/FP32 权重可独立于激活类型。
`.to(dtype=...)` 支持 CPU FP16/FP32 间转换并触发重新打包；BF16 和设备迁移明确拒绝。
非连续布局在确有需要时按原 dtype 做 contiguous，窗口 SDPA 可直接读取通道连续的 FP32 strided 视图。
无 backward/autograd；应在 `torch.inference_mode()` 下使用。注册了 FakeTensor 形状函数，
手动 NpuModule 的通用 compile/export 不作承诺；自动图适配请使用上面的专用 `make_backend`。

## 按工作量均匀分核

TFLLM 默认启用自适应分核（`NPU_SPLIT revision=5`），切分模型将 0–7 号 TFACC
视为相同算力。Linear（含 1×1 Conv）按每个至多 1024 行的 GEMM，
空间 Conv 按单图，融合 SDPA 按单个 head 的 QK+AV 工作量选择计算核数上限：

| MAC 数（默认门槛） | 上限 |
| --- | --- |
| ≤4×1024² | 1 核 |
| ≤16×1024² | 2 核 |
| ≤64×1024² | 4 核 |
| 更大 | 8 核 |

GEMM/Conv 比较各核等算力下的最长分片、IO 和核对提交成本，可能进一步减少核数。
对齐允许时均匀分配工作量，不采用大小核 2:1 权重；所有任务可在四个核对 worker
之间自由迁移。小算子减核控制并行分片数量，不指定只能使用哪几个物理核。
Conv 仍检查 128 像素分片粒度、halo 输入地址对齐和通道尾部。
SDPA 在工作量核数上限内，按 score 预算及 padding 限制选择 1–4 个核对，均匀分配
query band；每个核对内 QK/AV 也可以只启用一个计算核。Q1024/KV1024/D128、4 MiB
score 预算时，四个 band 各为 256 行，共执行 1024 行，不再因大小核权重多执行行。
实际命令仍分别编译 16/8-MAU 变体，由执行 worker 选择匹配硬件格式的版本；
这用于正确执行，不参与切分算力权重。

这些是可调的成本启发式，尚未在板端校准，不承诺每种 shape 都更快。
`TFLLM_NPU_SPLIT_MIN_MACS` 默认 `4194304`，范围 1..1073741824，可调整门槛及
提交成本估计。`TFLLM_NPU_MAX_CORES=1/2/4/8` 限制每个 GEMM/Conv 图/attention head；
并行 head 或独立请求仍可占用不同核对。计算单核仍获取完整核对的驱动 lease，
芯片内存初始化也仍会短暂获取所有核对。
`TFLLM_NPU_ADAPTIVE_SPLIT=0` 恢复旧 SDK Grid 策略，默认上限 8。
`TFLLM_NPU_TRACE_SPLIT=1` 打印 GEMM/Conv 模板和 SDPA 的分片选择；测速时关闭此日志。
拆分参数进入磁盘 recipe 版本，避免加载旧布局；无需重新量化模型。

板端重编部署 `libtfllm-torch` 后，用固定随机输入对比热运行，不下载 checkpoint：

```sh
TFLLM_NPU_ADAPTIVE_SPLIT=0 python TFLLM/examples/torch_split_benchmark.py > split_old.log
TFLLM_NPU_ADAPTIVE_SPLIT=1 python TFLLM/examples/torch_split_benchmark.py > split_auto.log
TFLLM_NPU_MAX_CORES=4 python TFLLM/examples/torch_split_benchmark.py > split_cap4.log
```

安装后脚本位于 `tfllm/test/torch_split_benchmark.py`。它分别报告 Linear/Conv/SDPA
的首次调用、预热后中位数和原生 Torch FP16 耗时，以及量化输出相对 Torch 的误差。
诊断时可另开 `TFLLM_NPU_VERIFY_OUTPUTS=1`，性能比较时关闭检查/profiler/trace。
CPU 参考或假 SDK 的计时不能作为板端速度证据。本次修改不增加 Torch CPU 自动回退。

## 长行 Linear 与窗口 SDPA 的执行优化

Torch 窄 Linear（逻辑 K/N 均不超过 1024）自动扩大 M tile：N>=256 时为 4096，
其余为 8192；其他 shape 保留 1024。后端接受至多 8192 行的独立执行，
不改变 Session 已有的整段 prefill 调度。长行投影默认使用两条独立执行通道，
每条通道拥有私有 A/C、命令及 scope，量化和反量化可以与另一通道的 NPU 执行重叠。
所有通道完成或异常排空后，才归还 Torch 的借用张量。

FP32 Linear 自动使用原值量化及直接 FP32 输出；非连续输入仅在需要时做 FP32 contiguous。
显式模块仍可调用 `NpuLinear.forward_fp32(x)`，与直接 `module(x_fp32)` 的通道一致。
FP32 权重直接预量化为不可变 U8/scales 快照，保留外层参数类型与版本检测。
Torch 空间 Conv 默认缓存有界私有执行实例，热运行刷新输入和量化参数；
既有 prefill/encode scope 优先，内存压力可回收闲置实例。

非因果、Q/KV<=1024、D<=256、至少 32 个独立窗口/head 的 SDPA 使用窗口批处理。
Q/V head 数及维度必须一致；其他几何、GQA、因果及区间 mask 继续走已有 NPU 路径。
每核默认合并 32 组窗口的已重定位 SDK 命令。KV<=32、Q<=64 时，最多 8 个窗口
通过对角 K/V 块共用一个 GEMM，填补 64 列对齐空洞；各窗口的 Q/P scale 和 softmax
范围独立，仍不构造跨窗注意力。16/49/196/576 等非 64 的 KV 长度不会进入 softmax padding。
每个核对在同一 lease 下执行批量 QK、CPU softmax/量化和批量 AV；
入口只需一次整核 invalidate，各窗口仍保留输出发布/排空命令。
最多缓存八个私有窗口实例，总计不超过 workspace cache 预算的一半；按最近使用顺序
淘汰，K/V 每次刷新，并参与压力回收。避免前两个 shape 永久占据缓存、后续 shape 反复分配。
需要 SDK 已有的 prepared-command API；API 不可用时使用原来的 NPU 路径。

`NpuRuntime.attention(q,k,v)` 自动识别 FP32；`attention_fp32()` 是同一通道的显式入口。
FP16/FP32 窗口直接读取通道 stride=1 的 B/H/Q 视图。大 dense attention（Q 或 KV>1024，
Hq=Hkv）也直接读取交错 QKV，输出为逻辑 BHQD、物理 BQHD，后续合并 heads 无需完整转置拷贝。
FakeTensor/compile 返回相同布局；其他 strides/GQA 保留既有通道。FP32 的量化与输出均不经过 FP16。
FP32 softmax 利用归一化分母在 U8 codes 中抵消，将 exp 求和与量化合并，减少中间读写；
浮点运算次序变化可能使临界舍入相差一个 U8 单位。FP16 概率舍入边界保持原样。
普通、因果、GQA、区间与 CSR 块稀疏 attention 同样具备直接 FP32 通道。
已提交后的设备错误继续上报，不重放到 CPU。

诊断开关（每组对照单独启动进程）：

| 环境变量 | 默认值 / 含义 |
|---|---|
| `TFLLM_NPU_LINEAR_TILE_ROWS` | `0` 保留默认；1024/2048/4096/8192 可覆盖普通 Linear 的行块，包括宽投影 |
| `TFLLM_NPU_LINEAR_LAYOUT` | `xw` 默认；`auto` 按形状选择；`wxt` 交换静态权重和输入的物理 M/N；`wxt15` / `wxt20` 按 full/lite 的 3:2 / 2:1 分配输出通道 |
| `TFLLM_NPU_LINEAR_AUTOTUNE` | 默认 `0`；`1` 在首次遇到普通静态 Linear 形状时测量布局、核数、M/N 网格和行块，按完整原生调用的中位数选择 |
| `TFLLM_NPU_LINEAR_TUNE_ITERS` | 校准时每个候选的正式次数，默认 `5`，范围 `1..31`；编译和首次准备不计入候选计时 |
| `TFLLM_NPU_LINEAR_TUNE_CONCURRENCY` | 默认 `1` 按单次耗时；`2` / `4` 用相应数量的并发原生调用校准吞吐；不改变服务的调用线程数 |
| `TFLLM_NPU_LINEAR_PIPELINE` | Torch NPU 默认 `1`；`0` 串行 |
| `TFLLM_NPU_CONV_WORKSPACE_REUSE` | Torch NPU 默认 `1`；`0` 关闭跨调用 Conv 实例缓存 |
| `TFLLM_NPU_WINDOW_BATCH` | Torch NPU 默认 `32`，范围 0..64；`0` 使用逐 head 路径 |

`NPU_OPERATOR revision=2` 打印 Linear 行块覆盖值、布局、Conv 缓存和窗口批量配置。
普通布局保留迁移调度；加权布局优先交给对应 full/lite worker，忙时仍可迁移。
重新构建原生库并同步 Python 包后运行：

```sh
python TFLLM/examples/torch_operator_benchmark.py --repeat 15 --threads 32
TFLLM_NPU_LINEAR_TILE_ROWS=1024 TFLLM_NPU_LINEAR_PIPELINE=0 \
  TFLLM_NPU_WINDOW_BATCH=0 TFLLM_NPU_CONV_WORKSPACE_REUSE=0 \
  python TFLLM/examples/torch_operator_benchmark.py --repeat 15 --threads 32
```

Linear 布局开关在创建 runtime 时读取，只用于普通静态 Linear；attention 和 Conv
分别使用各自的执行几何。`wxt` 计算 `W @ X.T`，CPU 将 INT32 结果转置并按原有 scale
反量化到 FP16/FP32，保留原量化精度。它对短 M 的部分大权重矩阵有收益，对大 M
可能因转置和反量化而变慢，不能把裸 NPU TOPS 当作完整 Torch 调用的收益。
`auto` 是可选的保守策略：M≤64、K/N 均≥4096、量化权重至少 32 MiB 时选择 WXT，
其余保留 XW。首次采用另一布局需要额外打包，同一权重混用长短 M 时会缓存两种格式。
行块也按完整调用评估；1024 行上限仍为宽矩阵的默认值，显式扩大不保证加速。
可选联合校准在 runtime 创建前开启：

```sh
TFLLM_NPU_LINEAR_LAYOUT=auto TFLLM_NPU_LINEAR_AUTOTUNE=1 python3 your_torch_program.py
```

这会在 XW/WXT、full/lite 比例、1/2/4/8 核、M/N 网格和行块之间搜索。
每个候选使用原量化参数和 INT32 输出，并与当前输入的基线完整输出逐字节核对。
选择按逻辑 M/K/N、FP16/FP32 和输入行距保存在当前 runtime 内；相同形状换权重仍可复用选择。
`NPU_LINEAR_TUNE` 日志给出布局、网格、行块、核数及候选计时。显式指定 `xw` / `wxt` 等
会限制搜索方向，`TFLLM_NPU_LINEAR_TILE_ROWS` 限制最大行块。

校准只用于通道连续、完整输入/输出、64 对齐且 K/N≤16384、M≤8192 的静态 Linear；
其他视图、共享量化输入的 FFN、attention 和 Conv 保留原执行路径。
候选通过现有 worker/pair 租约执行；编译拒绝和健康状态下的容量不足可跳过，
硬件错误或数值不一致立即上报。校准不绕过 NPU 锁。
首次搜索有额外时间和内存开销：板端重点形状约 1～7 秒，4096×4096×12288 的搜索约 60 秒；
固定形状服务应在接收请求前预热。选择本身不跨进程保存，磁盘 recipe 缓存另行配置。
并发服务可在创建 runtime 前再设 `TFLLM_NPU_LINEAR_TUNE_CONCURRENCY=2` 或 `4`。
校准使用持续存活的调用线程和独立输出，线程创建在计时外；目标是整个并发组的耗时除以完成调用数。
每个输出仍逐字节核对，所有任务结束后才传播错误。该选项会增加校准时间和临时输出内存。
它不会创建常驻服务调用线程，也不会修改现有 NPU 租约；实际请求的 QPS 仍需单独验收，
不能保证所有形状或所有线程数都更快。
对照数据与脚本见 [Linear 形状扫描](artifacts/torch-linear-tuning-20261019-server65/REPORT.md)。

脚本同时报告 FP16 NPU 入口、直接 FP32 入口和相同 FP16-rounded 操作数的 Torch FP32 基线，
保留各次样本及输出哈希；4 GiB 测试池默认通过 `--memory-fraction .125` 指定。
这些优化不意味着每个 shape 都超过 CPU：小窗口、小空间 Conv 和小 KV 交叉注意力
仍需按完整调用时间验证；设备计时和 CPU 参考实现都不能替代这个对照。

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

### SAM 等视觉模型的混合 attention 策略

小窗口和小 KV 的交叉 attention 在 A77 上仍可能由 Torch CPU SDPA 更快。
可以显式启用混合策略，同时保留 Linear/Conv 的既有路由：

```python
runtime = NpuRuntime(attention_policy="hybrid", attention_fuse_workers=32,
                     attention_score_mib=16)
# 将 runtime 交给 optimize()/make_backend()/tfllm.vision.load()；
# 手动 NpuAttention(runtime=runtime, implementation="auto") 也使用该策略。
```

`attention_policy` 默认 `"npu"`，也可通过 `TFLLM_TORCH_ATTENTION_POLICY` 设置；显式参数优先。
`"cpu"` 在高层始终选择原生 `F.scaled_dot_product_attention`；`"hybrid"` 目前只将非因果、无 mask、
Hq=Hkv、Q/KV 均>=1024、Q×KV>=4194304、48<=D=Dv<=256（FP16 为 32 起）、KV<=16384 的大 attention 交给 NPU，
其余 FP32 使用 Torch SDPA。外部 FP16 额外将 Q=16..1024、KV=32..1024、至少32个 batch/head、
总计 B×H×Q×KV×D>=16777216、同样无 mask/非因果/相同 heads 及 D 范围的窗口交给 NPU：Torch 2.14.1 的 FP16 SDPA
在这些形状上明显慢于 FP32。autocast 按实际执行 dtype 选择。该门限来自 A77/NPU40T、
Torch 32 线程测量，其他 CPU/线程配置需重新验证，
不是在线 autotune。选用 CPU 的调用在 optimization_report 中记为 `cpu_sdpa`，
整个官方 SAM attention 模块保留原实现时记为 `cpu_attention`；均不是 `fallback`。
`runtime.attention()`、`attention_fp32()` 和 `implementation="fused"` 始终强制 NPU，不受该策略影响；
设备错误继续直接上报，绝不提交失败后回放 CPU。此策略仅用于 Torch，不修改原生 LLM serving。

官方 SAM2Base/Sam3Image 使用模型内的 attention scope，避免整段编码的 TorchFunctionMode
拦截开销。Hiera 保留原 QKV、q_pool 和 proj，直接调用路由；SAM3 的 RoPE/relative position
仍执行原 forward，只有需要路由的模块进入 scope。已确定走 CPU 的 FP32 模块直接执行原实现。
scope 仅在 VisionSession 方法调用中激活；直接调用原 predictor 仍可作为 Torch SDPA 对照。
自定义实例 forward、未识别的直接 SDPA 或其它模型保留已有的 session scope。

### 不连续块掩码（FlashVSR 等专用 attention 的计算核心）

新增显式 `NpuBlockSparseAttention`，支持不同 query head 各自选择多个不连续 KV 块。
每个 query 块在 C++ 收集所选 K/V，再调用一次原生融合 QK → softmax → AV；
**softmax 在该 query 块选中的全部 key 上统一归一化**，不是独立算各段再平均。
不生成完整 token 级 mask、QK 或 probability，不走 Python 分块 / Torch softmax。
沿用现有八核调度、核对流水、score tile 预算和命令模板重定位，没有修改 NPU 驱动。

```python
from tfllm.torch import BlockSparseMask, NpuBlockSparseAttention

# q/k/v 是同 dtype 的 CPU FP16/FP32 [B,H,L,D]；先由模型完成 RoPE、窗口重排、KV cache 拼接。
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

空间卷积可用独立板端测试复现，不需要下载 checkpoint：

```sh
PYTHONPATH=TFLLM/python TFLLM_TORCH_LIBRARY=/path/to/libtfllm-torch.so \
  python TFLLM/examples/torch_conv_probe.py --repeat 5
# 同一 stride=4、同一输出 shape，加入 1x1/3x3 对照：
PYTHONPATH=TFLLM/python TFLLM_TORCH_LIBRARY=/path/to/libtfllm-torch.so \
  python TFLLM/examples/torch_conv_probe.py --controls --repeat 5 --dump conv_failure.pt
```

默认输入 `[1,3,1024,1024]`、输出通道 96、7x7/stride4/pad3，输入范围约 [-2.07,2.64]。
测试固定输入和权重，包含第一次冷调用，不自动预热；打印量化参考与浮点参考的误差、
NPU 与量化参考的误差，以及重复调用的逐 bit 一致性。`--dump` 仅在失败时保存第一组
输入、权重、参考和实际输出；随机权重通过不能证明真实 checkpoint 已修复。
`--replay conv_failure.pt` 可直接复用保存的输入/权重/bias/kernel/stride/pad，
无需依赖随机数生成器或模型加载；再加 `--stride 2` 可只改变 stride 做对照，
量化参考随新几何重新计算。replay 不与 `--controls` 同时使用。

新构建可另外设置 `TFLLM_NPU_VERIFY_OUTPUTS=1`：Linear/融合 attention 沿用原 INT32
抽样核对，空间 Conv 新增每个输出通道至多 16 个位置的 INT32 核对，覆盖角点和内部。
Conv 核对使用 canonical U8 权重，不把 DMA-packed 权重作为参考；在同步完成后、
workspace 归还前检查，失败抛出 `NPU_CONV_I32_MISMATCH`，含 core、shape、通道、
位置、zeroPoint 与 expected/actual INT32。默认关闭，抽样不保证发现所有错误。

有三个独立的 CPU 并行设置：

```sh
# 通用量化/反量化及其它 cpu::Rows 内核，1..32，默认 8。
# 在首次 TFLLM CPU 调用前设置；每个 A/B 配置单独启动进程。
TFLLM_CPU_THREADS=32 python your_sam_demo.py
```

```python
import torch
from tfllm.torch import NpuRuntime

torch.set_num_threads(32)  # 保留的 Torch 算子，与 cpuLimit 无关
runtime = NpuRuntime(attention_fuse_workers=32)  # 融合 attention 的 CPU 工作
```

`TFLLM_CPU_THREADS` 包含调用线程；共享 helper 池只增不减，显式 `RowsLimited`
仍遵守其自身上限。小工作量保持串行，不保证每次使用指定数量的线程。
启动日志 `cpu_row_workers` 显示通用上限，`fuse_workers` 显示 attention 上限，
`cpu_background_threads` 是共享池的 helper 容量，不是每个算子的参与线程数。
三个设置不互相替代，也不控制四个 NPU 核对管理线程。板端分别测 8/16/32，
以关闭 profiler 后的端到端时间选配置；线程更多不保证更快。

CPU 参考路径的独立回归探针（不需要 NPU 或模型权重）：

```sh
TFLLM_CPU_THREADS=32 python TFLLM/examples/torch_cpu_reference_benchmark.py --repeat 3 --warmup 1
```

它分别计时同一组可精确表示为 FP16 的数据在 FP16/FP32 入口上的运行时间，
同时报告 Torch FP32 对照和误差；权重准备、dtype 转换不计入算子时间。
空间 Conv 的 CPU 模式执行仿射 U8/I32 参考卷积，和浮点 Linear 参考的计算语义不同。

`attention_fuse_workers` 为 1..32（默认 16，包含核对调用线程），共享 CPU 线程池只增不减；
线程数 A/B 应分别启动进程。`attention_score_mib` 为 1..16（默认 4），是一个 slot 跨四个
pair lane 的 INT32 score 预算，双缓冲使用两个 slot，最小 16 行对齐可能向上取整。
它不是整个 runtime 的内存上限；还有 K/V、UINT8 激活、AV 输出和命令缓冲。
Torch 与原生 vision 共用最多两个并行 head job，单次调用最多提交两个，异常时全部等待结束。
`attention_lease_tiles` 为 1..16（默认 4），控制一个核对连续处理多少 tile 后归还 lease。
小形状可能只启用一部分核对或单缓冲，不保证八核始终满载。

大 attention 的双缓冲默认启用 `TFLLM_NPU_ATTENTION_QK_OVERLAP=1`：在下一块
QK 执行期间完成前一块 softmax，并在消费旧 Q scale 后准备后续 Q；PV 期间反量化
已经完成的输出。保留完整 INT32 QK 和原有舍入语义。设为 `0` 可恢复原调度做 A/B。

`TFLLM_NPU_ATTENTION_QK_SHIFT=0` 默认保持 INT32；显式设置 `8..16` 可试验直接
UINT8 QK 输出，按 `(code-128)*2^shift` 恢复整数累加值，再使用原有 Q/K scale。
该模式只作用于 padded head dim≤256 的 dense fused 路径，需要本次配套的 NPU40T SDK；
窗口批处理保持原实现。检测到 code=0/255 的行按 16 行一组在 NPU 重算 INT32，
若至少四分之一行组触边，该 head/lane 的余下部分使用 INT32。设备错误仍按原策略报错，
这里的重算仅处理数值饱和。未饱和的 U8 行仍有量化误差，shift 越大舍入越粗，不能视为无损。
运行时保留 INT32 重算工作区，因此主要减少正常 QK 写回量，不降低 score 工作区峰值。
`TFLLM_NPU_VERIFY_OUTPUTS=1` 可在计时外将 U8 与设备 INT32 重定标结果逐元素核对。
板端精度、性能和真实 SAM 单图掩码证据见 `artifacts/attention-qk-20261016-server65/REPORT.txt`。
真实 Qwen3-VL-8B 文本分支补测见 `artifacts/attention-qwen3-20261016-server65/REPORT.txt`。
该模型的采样 QK 在固定 shift=11 下未出现端点，但完整文本分支测试仍显著改变
token 分布；饱和回退不能检测这类舍入损失。当前固定 shift 的 U8 路径尚不能
作为 LLM 的通用精度方案，LLM 保持默认 INT32。
后续 K/V 去均值与 Q/K 正交旋转的独立精度原型见
`artifacts/attention-precision-20261016-server65/REPORT.txt`。两个短文本的真实
Qwen 文本分支测试有改善；含预处理成本的性能依形状而异，尚未接入生产默认后端。
旋转与量化进一步融合的隔离原型见 `artifacts/attention-fused-20261016-server65/REPORT.txt`：
2048 上下文五层 attention 样本约 1.46x，512 仍变慢；不是整模型吞吐，默认路径不变。
硬件 MAX/SUM 生成行统计、CPU 只处理标量概率尺度、硬件 softmax P8 接普通 PV
的后续原型见 `artifacts/attention-hwsm-20261016-server65/REPORT.txt`。
四个 full 核的小 tile 命令链相对同输入 CPU P7 对照约 1.085x；较短 KV 和大
query tile 仍慢，尚未证明完整 TFLLM API 有额外收益，也未接入默认路径。
多命令队列的后续排查见 `artifacts/attention-multiqueue-20261017-server65/REPORT.txt`：
现有 runtime 固定使用 queue0；隔离探针已验证 queue0/1 的独立传输与完成检测，
计算命令的 queue1 上下文/完成路由仍待确认，尚无有效的 GEMM/softmax 并发加速数据。

`TFLLM_NPU_ATTENTION_FAST=1` 现为主程序的通用可选 attention 算法，创建
NPU runtime 时读取，默认关闭。它已移除 Qwen 的 head 数、causal、2048/4096
profile 门槛；FP16/FP32 dense、strided、window、interval、block-sparse，以及
原生 KV-cache attention 共用 `src/fast_attention.cpp`。支持 batch、GQA、cross
attention、单 query 和任意非空长度，QK/value head dimension 均须 ≤256、有效
keys ≤16384。这个限制没有扩展到普通 Linear/FFN 的 U8 输出。

```sh
TFLLM_NPU_ATTENTION_FAST=1 tfllm serve /path/to/model --backend npu
```

```python
from tfllm.torch import NpuRuntime, optimize

runtime = NpuRuntime(attention_fast=True)  # 显式 bool 优先于环境变量
optimize(model.eval(), runtime=runtime)   # 模型 forward 内的 SDPA 也受此选择控制
# with torch.inference_mode(): output = model(...)
# 或直接 runtime.attention(q, k, v, attn_mask=mask, ...)
```

`attention_fast=None`（默认）继承环境变量；`False` 可覆盖环境中的 `1`。
模式在创建时确定，`runtime.attention_fast` 只读。Torch 的 `attention_policy`
控制默认算法的 CPU/NPU 分流，FAST 开启后进入 TFLLM 的 attention 统一选择 NPU
FAST，不再被 hybrid 的短窗口门槛排除。支持任意 bool 可见性、逐 head 浮点
additive bias、causal 与 mask 组合和全屏蔽行（输出零）；广播 mask 保留原 strides，
不展开为完整 FP32 score。仅有区间的 mask 保持紧凑表示。

Torch optimize/vision 的 scope 接管其中调用的 SDPA，packed Torch/SAM3 MHA
保留原投影、mask/bias 和布局。`need_weights=True`、训练/dropout、自定义未适配
attention 或超出上述维度明确报错；显式 FAST 不会悄悄退回旧 attention。
任意手写 `matmul → softmax → matmul` 代码仍需适配器或图后端接管。

serve 保留 canonical KV cache（F16 和默认 Q8_0），每次仅为当前有效前缀创建
临时变换/量化快照，不改写共享 cache。截短、前缀复用、追加、soft cap、sliding
window 使用相同 FAST 内核；soft cap 下停用 K 去均值以保留其非线性语义。
**保留现有后端分工：llama.cpp CPU 单 token decode 仍由 CPU 执行，Linear/FFN
不因开启 FAST 而迁移。** 原生 NPU attention 被调用时才使用该算法。

计算为 K/V 去均值（适用时）与 Q/K 有符号正交 Hadamard 旋转，Q 按行、K 按
key 行、V 按通道量化。D 非 2 的幂时使用独立正交子块；物理 padding 不参与
逻辑 head 的 scale。原来的 1/sqrt(D) 或调用者 scale 仍在 softmax 前应用。
QK 在芯片内 INT32 累加后直接重定标输出 UINT8（shift=11，优先于 QK_SHIFT），
保留饱和行组的 NPU INT32 重算。CPU 按 query tile 读取 U8，行 scratch 中计算
FP32 softmax 并生成 U8 P7，随后 NPU PV 输出 INT32；反量化加回 V 均值后再转
外部 FP16/FP32。没有增加完整 FP32 score/probability 矩阵，未使用硬件 softmax。

数学变换在量化前等价，UINT8 QK/P7 仍有量化误差；饱和保护不检测全部舍入
误差，因此默认关闭。短窗口的变换/调度成本可能超过带宽收益，FAST 名称不承诺
任意 shape 都更快。启动日志 `NPU_ATTENTION fast=1 ... qk_u8_shift=11` 与 Torch
coverage 的 `npu_fast` 可核对启用情况。C ABI2 保持兼容，新增 feature version/
构造及 mask 入口；内部 C++ 接口需要整体重编译，SDK/Python 应配套更新。

通用路径的开发机验证、真实模型覆盖和安装记录见
[通用 FAST 报告](artifacts/attention-general-fast-20261017-server65/REPORT.md)。
下面保留此前限定 Qwen profile 的历史性能数据，不能外推到新覆盖的形状。

10月17日后续 softmax 优化已写入源码：CPU 行任务成组入队，小组只唤醒所需
数量的 helper，U8 分数按
16 元素解码；默认保留原 exp、sum 顺序和 P7 舍入。真实 2048/4096 API 及
一个 2048-token、36 层原生模型的最后位置 logits 与原 FAST 逐 bit 一致。
默认 attention 参数下 FP16 API 时间分别约减少 6.8%/8%；真实2361-token服务
两轮TTFT收益波动较大，详见新报告。这是独立测量，不能与下文历史数据混算。

`TFLLM_CPU_SOFTMAX_POLY4=1` 是额外的实验开关，首次使用 FP32→P7 CPU
softmax 时读取，默认 `0`。它启用四阶 exp 和独立累加，可进一步降低时延，
但会改变 P7 阈值附近的舍入和行尺度；一份完整模型最后位置诊断中 KL 变差，
NRMSE/余弦变好，尚未做任务集/PPL 验证。它也作用于内部使用 FP32 softmax
的 FAST FP16/native attention；通用 activation exp、显式 FP16 概率舍入及
soft-cap 分支不受该开关影响。非 NEON 平台仍使用 scalar exp。
本轮板端测试使用私有重链接镜像，正式安装 SDK 尚未替换；源码需重新构建后生效。
方法、消融、线程扫描和完整模型证据见
`artifacts/attention-softmax-20261017-server65/REPORT.md`。

当前完整 API 复测包含全部预处理：2048 的三个真实采样层从约 26.7～27.0 ms
降到 17.9～18.0 ms；4096 第 0 层从 85.0 降到 60.0 ms。此前未进入 FAST 的 512 输出与原 I32
逐 bit 相同；通用 FAST 现在也会处理该长度。计时配置为 `attention_fuse_workers=32`、`attention_score_mib=16`、
`memory_fraction=.125`。测试方法、模型数值对照与可复现脚本见
`artifacts/attention-fast-20261017-server65/REPORT.txt`，这些不是整模型 tokens/s。

FP16 与真实服务验证见 `artifacts/attention-serve-fast-20261017-server65/REPORT.txt`。
最终版本在默认 attention 参数下，对一组 2349-token 请求测得 TTFT 中位数
10.52s → 7.74s（各2个测量样本、另各2次预热）；前缀复用、追加和两请求并发通过。
该数据来自 Qwen3-VL-8B 的文本模式，不能外推为任意上下文的稳定加速比。

INT32 ReduceMax 已在 full/lite 共八核通过；UINT8 限制来自上层 ReduceHelper，
底层可通过 `REG_POOLING_INPUT_INT32` 切换。cache 内 QK/MAX/LUT/SUM/P16/PV
完整实验见 `artifacts/attention-cache-lut-20261017-server65/REPORT.txt`。
其完整 2048-token API 约 30～31 ms，尚慢于上述快路径，未纳入该策略。

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

Torch runtime 默认使用按需增长的物理内存池，使用 TFLLM 原有四个核对线程管理八个 TFACC。
独立进程可以同时创建 runtime、加载模型和提交推理，由驱动核对锁协调执行。
`memory_fraction=0.8` 表示每个 runtime 的内存上限，按注册大页总 GiB 数取整；
32 GiB 注册量对应 25 GiB 上限，启动时实际占用为 0，而非立即抢占 25 GiB。
模型权重和执行 workspace 在同一个 HIGH 内共享空间，不再强制按地址低 3 GiB／高 1 GiB 分区。
申请按驱动大页块增长（通常 1 GiB）；驱动以整块注册大页划分进程所有权，小分配也可能独占整块。
完全空闲的段会归还驱动，供其它进程使用；在途命令、常驻权重和仍缓存的实例不会被提前释放。
可用 `runtime.memory_stats()` 查看 `registered_bytes`（芯片注册总量）、`budget_bytes`（本池上限）、
`committed_bytes`（本池已申请的大页段字节）及 `used_bytes`（活跃分配和直接命令缓存）。
驱动整页所有权占用可能大于 `committed_bytes`，直接命令内存仍来自独立 reserved-DDR 域。
总物理容量、单 HIGH 连续空间不足时仍会明确报错，各进程的上限之和不代表物理容量保证。
本项需要同步重建 SDK `libNPU40T` 和 TFLLM；`TFLLM_NPU_ELASTIC_MEMORY=0` 可恢复固定预留模式。
原生 LLM serve 默认继续固定预留，Torch 的新默认不修改 serving 内存策略。
GEMM 默认按工作量和等算力模型切分 M/N；原生 Conv 按输出通道与高度分片，包含正确输入 halo。
高度分片要求每个输出行带的 `Hs × Wout` 都是 128 的倍数（同时满足 64 对齐），
并保持 halo 后输入起始地址 4 字节对齐；无法完整覆盖时只切输出通道，不切高度。
例如 `140×140`、`172×114` 的 `3×3 / stride=1 / pad=1` Conv 保留整幅高度，按通道和工作量决定分片数。
这不改变逻辑 H/W，也不进行图像 padding；整幅及分片缓冲仍使用安全的物理通道间距。
每个分片准备 16/8 MAU 变体，任务可在任意空闲核对之间调度。能否同时用满八核取决于形状和竞争，
不会为不足以有效拆分的小算子伪造八路工作。多个 module 不能各创建一套 runtime。

静态权重与在途工作空间分离；并发执行命令和 A/C 缓冲私有，完整等待后回收。
错误/超时沿用原有芯片隔离，不释放可能仍被 DMA 使用的资源。close() 等待本 runtime 的活动调用结束。
state_dict 保存 CPU 权重，不保存物理地址；加载后显式 prepare() 或首次 forward 重建 packing。
普通原地权重修改通过 PyTorch version counter 检测；禁止 `.data` 绕过版本机制及并发修改/推理。
模块 close() 释放其 packed handle，下一次 forward 可重建；runtime close() 则永久关闭全部模块。

当前保留 CPU 权重副本以支持 checkpoint，不能把常驻占用简单等同于 INT8 权重大小。
原生 Conv 权重只准备一次；命令采用与 MatMul 相同的 **CPU 不可变模板 + 独占可执行实例**。
首次遇到卷积几何形状时编译 8/16 MAU 两套配方，缓存按形状而非权重、输入 zero-point 或物理 HIGH 区分。
实例未命中时复制命令镜像并重定位输入/输出/权重/LUT/临时缓冲地址；命中时复用已绑定命令。
每次调用仍写入新输入、量化参数和动态 K/V。静态 Linear 切换同形状权重时只更新 B 地址，
Conv 更新 padding/bias 中的 zero-point；不缓存计算结果或调用者的输入张量。
原有原生卷积形状限制不变；例如部分超宽 stride-2 形状仍受 SDK 输出地址对齐限制。
没有板端吞吐、延迟和模型质量实测前不承诺性能。

Torch NPU 默认开启跨 forward 命令实例缓存。Linear、Conv/ConvTranspose2d、大 Attention 和窗口
Attention 只把执行完成且 CPU 读取结束的实例放回空闲池；并发调用独占 checkout，绝不共用在途 A/C、
K/V 或可变命令。缓存属于 runtime，按几何、核数、cache 策略、动态权重类型和兼容 HIGH 区分，
不会按输入 Tensor 地址缓存输出。

- `TFLLM_NPU_FORWARD_CACHE=1`：默认开启。`0` 恢复请求作用域 Linear/大 Attention，保留原有 Conv/窗口缓存。
- `TFLLM_NPU_WORKSPACE_CACHE_MIB=1024`：整个 forward 空闲执行缓存共用的预算，包含 A/C、scratch、
  命令以及私有动态 K/V，最多 32 个实例；窗口仍另有限额八个、总预算一半。超额按最近使用顺序淘汰。
- `TFLLM_NPU_TEMPLATE_CACHE_MIB=64`：CPU 命令镜像预算，Linear/Conv 合计最多 512 份模板。
  活动或空闲实例仍可持有其不可变模板，直到实例释放；此上限不包含这些引用和 SDK 元数据。
- `TFLLM_NPU_CACHE_IDLE_MS=60000`：空闲超过 60 秒由后台线程回收实例和不再使用的模板，
  无需下一次 forward 触发，检查间隔至多 1 秒。`0` 关闭定时过期，保留容量淘汰。
- `TFLLM_NPU_WORKSPACE_REUSE=0`：关闭跨 forward 实例缓存和普通 GEMM/Conv 工作区复用；
  Attention 的请求内复用另由 `TFLLM_NPU_ATTENTION_WORKSPACE_REUSE` 控制。CPU 模板仍可复用。

选项在创建 runtime 时读取；close() 停止回收线程并清空缓存。内存不足时优先回收空闲实例，
错误/超时保留原 quarantine：不回收或重放可能仍在运行的命令。释放的缓冲区先归还 SDK 池；
按需池中完全空闲的段进一步归还驱动。固定预留模式和已增长的直接 command pool 继续保留容量。原生 LLM serve 默认仍采用
请求内复用，请求结束归还实例，不启用此跨 forward 缓存。

多进程板端复现（默认内存上限，随机固定操作数，无需模型下载）：

```bash
TFLLM_NPU_VERIFY_OUTPUTS=1 python3 examples/torch_multiprocess_probe.py --workers 3 --repeat 10
```

板端 A/B 测试（不同进程，同一库、输入、线程数；包含 cold 与 warm median）：

```sh
TFLLM_NPU_FORWARD_CACHE=0 python3 TFLLM/examples/torch_command_cache_benchmark.py
TFLLM_NPU_FORWARD_CACHE=1 python3 TFLLM/examples/torch_command_cache_benchmark.py
```

## NPU stalling / 超时诊断文件

原生 runtime 在核对准备、提交、轮询、cache handoff 或释放阶段抛出错误，以及轮询超时
（默认 30000 ms）时，先隔离芯片并保留在途 DMA/lease，再尽力导出当前核对的现场。
正常运行不写文件；Conv、Linear 和融合 attention 的底层 GEMM 共用这条路径，不要求开启 profile。

```sh
# 在测试包目录运行，不需要修改 Python 脚本：
TFLLM_NPU_DUMP_DIR="$PWD/npu-debug" python test_tfllm_npu.py \
  --backend npu --mode all_tfllm --samples woman --output results/woman.json
```

默认目录是进程工作目录下的 `./tfllm_npu_dumps/`；目录不存在时创建。
`TFLLM_NPU_DUMP_DIR=0` 关闭导出。C++ 也可用 `NpuOptions::failure_dump_dir` 覆盖，
环境变量在 runtime 初始化时读取。启动输出 `NPU_CONV split_revision=3 layout_revision=3 ... split_spatial_alignment=128 input_base_alignment=4 ...`
可确认已加载修复版本。

故障时 stderr 输出 `TFLLM_NPU_FAILURE_DUMP path=...`。文件名包含 PID/chip/pair 和随机后缀，
不覆盖已有文件，权限为 0600；多个核对失败可能各生成一个文件，请一并保留。文本内容包括：

- 原始错误、阶段、执行核、超时配置、提交/完成状态。
- Conv 的完整几何、batch 序号、输入量化参数、全部分片的 halo/偏移/stride；GEMM 的 M/K/N。
- 当前核对执行的分片、8/16 MAU 变体、命令 DMA 地址、operand HIGH、命令数量及十六进制命令字。
  每核最多输出 1 MiB 原始命令对应的文本，超出会标记 `truncated=1`。

只读取 CPU 持有的不可变命令与元数据，**不读取可能仍被 NPU 写入的输出/工作区，不执行额外 cache
维护或寄存器读写，也不导出原始激活/权重**。仍应按内部调试资料处理这些文件。
这不是完整数值回放包，也不包含 Python 层名；可用几何和分片定位算子。
写文件失败会打印警告，但不会替换原始错误、自动重跑或释放隔离资源。
若进程被强制终止/断电，或 SDK 调用本身永久阻塞而不返回，不能保证导出。

本次新增 SDK 只读接口 `tfaccGetCommandDebugInfo`，**必须一起重建安装 `libNPU40T`、TFLLM
及 `libtfllm-torch`**。pip 使用者还需从新的 CMake 安装树重新安装 wheel/包，避免旧库与新代码混用。

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
`tfllm-spatial-split` 覆盖 64512 组行切分几何；真实命令测试检查 woman 分片不会污染输入控制位，
并拒绝未对齐模板。原生注入测试对 woman 的 3→48、48→48、48→12 卷积做完整未分片量化参考对比。
`tfllm-failure-dump-*` 覆盖持续 BUSY、驱动异常、错误状态、不可写路径、禁用导出和 GEMM，验证
文件字段/权限、原错误传播、在途资源隔离和禁止失败后重放。
板端仍需对比 PyTorch 整模型误差，并分别测量 packing/量化/命令准备/NPU/dequant/CPU 算子的耗时。

本次主机验证使用 macOS arm64：独立构建及 CTest、ASan/UBSan、安装后自动找库均通过。
SDK 内 NPU40T + tfllm-npu40t + tfllm-torch 的编译链接也通过（GCC 15，复用已有 build/Release 生成头文件）。
这不等于板端硬件执行验证；本机没有 TFACC。原有 SDK 全量目标在 AppleClang 下还存在
InnerProductHelper 的 float/double 模板匹配问题，与本适配层无关，本次未修改。

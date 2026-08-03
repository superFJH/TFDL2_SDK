# ViT / Transformer 的 TFDL2 双图直接构建与快速量化

本文记录 [`ConvertTools/python/example/Vit.py`](../ConvertTools/python/example/Vit.py) 的实现方法，目标是为后续 ViT、DINO、LLM 和多模态 Transformer 的直接转换提供一套可复用方案。

这套方案不依赖 ONNX 还原部署图，而是同时维护两张拓扑等价的计算图：

- **TFDL 图**：使用 `TFDL2.Op` 直接构建，面向 `.fb`、`.quant.fb` 和 NPU 部署。
- **PyTorch 等效图**：使用相同权重、相同算子顺序和相同量化边界，负责与原始模型对拍，并在 GPU 上快速收集激活范围。

PyTorch 输出的 JSON 在当前实现中保存的是各中间 tensor 的 `min/max`，下文统一称为 **range JSON**。文件可以命名为 `quant.json`，但它不是最终量化权重，也不直接保存 scale/zero-point；最终权重量化和量化图改写仍由 TFDL2 SDK 的 `TFCalibration.Quantize` 完成。

## 1. 为什么使用双图直接构建

通用 ONNX 转换适合拓扑稳定、算子能够被转换器完整识别的模型。但 Transformer 导出后经常出现以下问题：

- 同一种 Linear 被导出为不同形态的 `MatMul`、`Gemm`、`Add` 和 `Reshape` 组合。
- QKV、Attention 和 MLP 周围插入大量 `Transpose`/`Reshape`，优化器不一定能恢复原始语义。
- 自定义 RoPE、特殊 Attention、Gated MLP 或远程代码模型不容易稳定导出。
- 即使转换成功，带权重 MatMul 没有改成 1x1 Conv 时也可能无法获得理想的 NPU 性能。
- 在 TFDL 软件执行器中逐样本运行大模型进行校准，耗时明显高于 PyTorch CUDA。

直接构图的代价是需要显式维护模型拓扑，但可以精确控制 NPU 上的算子、layout、量化边界和输出。PyTorch 等效图则解决了直接 TFDL 图不便快速校准与调试的问题。

## 2. 总体架构

```text
Hugging Face config.json + safetensors
                    │
                    ├── 解析统一 ViTOpConfig
                    │
                    └── 权重规范化 canonicalize_weights
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
      原始 PyTorch/HF 模型                  共享规范权重字典
              │                                   │
              │                     ┌─────────────┴─────────────┐
              │                     │                           │
              │                     ▼                           ▼
              │             TorchViTOpGraph             build_vit_tfdl_graph
              │             PyTorch 等效图                  TFDL 直接图
              │                     │                           │
              ├──── 数值对拍 ◄─────┤──── 数值对拍 ────────────┤
              │                     │                           │
              │                     ▼                           │
              │            GPU 收集逻辑 tag 的 min/max          │
              │                     │                           │
              │                     ▼                           │
              │                 range JSON                      │
              │                     │                           │
              │                     └── tag → TFDL symbol ─────►│
              │                                                 │
              │                                    AddInt8Config(max, min)
              │                                                 │
              │                                    TFCalibration.Quantize
              │                                                 │
              ▼                                                 ▼
         参考数值结果                                    .fb / .quant.fb
```

双图不是两套互不相关的实现。它们必须共享以下契约：

1. 同一份模型配置和规范化权重。
2. 相同的算子顺序、激活函数、Attention 缩放、残差位置和归一化位置。
3. 相同的预处理语义。
4. 相同且稳定的逻辑 tensor tag。
5. 允许 layout 不同，但 tag 对应 tensor 的元素值集合必须相同。

## 3. Linear 转 1x1 Conv

### 3.1 数学等价关系

Transformer Linear 的输入通常为：

```text
X: [B, S, Cin]
W: [Cout, Cin]
b: [Cout]
Y[b, s, o] = Σi X[b, s, i] * W[o, i] + b[o]
```

只要选择 `H × W = S`，就可以把 token 维重排为空间维：

```text
X [B, S, Cin]
  → Transpose [B, Cin, S]
  → Reshape   [B, Cin, H, W]

W [Cout, Cin]
  → Reshape   [Cout, Cin, 1, 1]
```

对该输入执行 `kernel=1` 的卷积，每个空间位置只进行通道投影，不会混合不同 token，因此与 Linear 数学等价。卷积输出再 reshape/transpose 回 `[B, S, Cout]` 即可。

`Vit.py` 使用 `seq_map_hw` 为完整序列长度寻找一组因子。该空间形状不要求等于原始 patch 网格，因为这里的 H/W 只承载 token 序列，1x1 Conv 不使用相邻空间关系。若序列长度只能分解为很瘦的形状，数值仍等价，但应单独评估 NPU 性能。

### 3.2 哪些投影应改成 Conv

| Transformer 模块 | TFDL 实现 |
| --- | --- |
| Q、K、V 权重投影 | 1x1 `Op.Convolution2` |
| Attention output projection | 1x1 `Op.Convolution2` |
| MLP fc1/fc2 | 1x1 `Op.Convolution2` |
| Gated MLP gate/up/down | 1x1 `Op.Convolution2` |
| Patch embedding | 保持原始 kernel/stride 的 `Op.Convolution2` |

下面两类乘法的两侧都是运行时 activation，不是固定权重投影，必须保留 MatMul：

```text
Attention score: Q @ Kᵀ
Attention value: softmax(score) @ V
```

不能以“模型里还有 MatMul”为失败标准。正确标准是：带固定权重的 Linear 投影已经使用 1x1 Conv，而 Attention 的 activation×activation MatMul 仍然存在。

### 3.3 layout 转换应集中管理

当前实现使用四个辅助函数隔离 layout 细节：

- `_tokens_to_conv1d`：`[1,S,C] → [1,C,H,W]`
- `_conv1d_to_tokens`：`[1,C,H,W] → [1,S,C]`
- `_qkv_conv_to_heads`：卷积输出转为 `[1,heads,S,head_dim]`
- `_attention_to_conv1d`：多头 Attention 输出转回卷积 layout

新增模型时应延续这个做法，避免每个 block 重复写不同的 reshape/transpose 序列。layout 变换一旦写错，最终输出可能 shape 正确但 token/head 顺序已经改变。

## 4. 统一配置和权重规范化

### 4.1 `ViTOpConfig`

`ViTOpConfig.from_model_path` 从 `config.json` 和可选的 `preprocessor_config.json` 中解析：

- 输入尺寸、patch size、通道数
- hidden size、层数、head 数、MLP 中间维度
- LayerNorm epsilon 和各种 bias 开关
- CLS/register token
- 绝对位置编码或 2D RoPE
- 普通 GeLU MLP 或 gated/SwiGLU MLP
- image mean、std 和 rescale factor

构图前至少校验：

```text
hidden_size % num_attention_heads == 0
image_height % patch_size_height == 0
image_width  % patch_size_width  == 0
使用 DINOv3 2D RoPE 时 head_dim % 4 == 0
```

当前 TFDL ViT 构图固定 batch 为 1、固定图像尺寸和固定序列长度。要支持动态 batch/shape，不能只修改 Placeholder，还需要检查所有显式 `Reshape`、位置编码和 RoPE 常量。

### 4.2 safetensors 加载

`load_safetensors` 支持：

- 单文件 `model.safetensors`
- `model.safetensors.index.json` 描述的分片权重
- 目录中的多个 `.safetensors` 文件

加载后统一转成 CPU、连续内存的 `numpy.float32`。TFDL 注册阶段保留 float 权重，由 SDK 在 `Quantize` 阶段完成权重量化，不要预先把主权重字典改成 uint8。

### 4.3 canonical namespace

不同框架的权重名称先映射到一套稳定的内部命名，再供两张图共同使用。核心命名如下：

| 规范名称 | 含义 |
| --- | --- |
| `patch.weight/bias` | Patch embedding |
| `prefix_tokens` | CLS 与 register token |
| `pos_embed` | 插值到目标尺寸后的绝对位置编码 |
| `layers.N.norm1.*`、`norm2.*` | Block LayerNorm |
| `layers.N.q/k/v.*` | 分离后的 QKV 1x1 Conv 权重 |
| `layers.N.proj.*` | Attention 输出投影 |
| `layers.N.fc1/fc2.*` | 普通 MLP |
| `layers.N.gate/up/fc2.*` | Gated MLP |
| `norm.*` | 最终 LayerNorm |
| `rope_sin/cos` | 预计算的 DINOv3 2D RoPE 常量 |

`canonicalize_weights` 同时处理：

- Hugging Face 和 timm 风格的多种权重前缀。
- packed QKV 权重沿输出通道拆成 Q/K/V。
- 缺失 bias 时补零。
- Linear 权重从 `[out,in]` 转为 `[out,in,1,1]`。
- 输入尺寸变化时对绝对位置编码执行 bicubic 插值。
- 预计算 DINOv3 2D RoPE sin/cos。

### 4.4 LayerScale 融合

如果原始 block 在投影后执行逐输出通道的 LayerScale：

```text
y = gamma * (W x + b)
```

当前实现将其提前折叠到 Conv 参数：

```text
W' = gamma[:,None,None,None] * W
b' = gamma * b
```

这样可以删除单独的运行时乘法，同时保持残差前的结果一致。新增类似参数融合时，必须在原始模型、PyTorch 等效图和 TFDL 图之间统一，否则 range 对应关系也会失效。

## 5. PyTorch 等效图

### 5.1 它不是原始模型的简单 wrapper

`TorchViTOpGraph` 使用 `torch.nn.functional` 按计划部署到 NPU 的拓扑重新实现 ViT：

- Linear 也用 `F.conv2d` 执行规范化后的 1x1 权重。
- QKV、Attention、MLP 的 layout 与 TFDL 图保持相同语义。
- DINOv3 只对 patch token 应用 2D RoPE，CLS/register token 保持不变。
- 使用与 TFDL 图相同的 canonical 权重。

因此它既是 TFDL 图的数值参考，也是量化范围采集器。不能直接在原始 Hugging Face 模型上随意注册 hook 后就把范围交给 TFDL，因为经过融合、layout 改写或算子边界变化后，两张图的量化节点不一定一一对应。

### 5.2 稳定的语义 tag

每个需要范围的 tensor 通过 `_record(tag, tensor)` 记录。`RangeCollector` 对多张校准样本取全局极值：

```python
qmin = min(previous_min, tensor.min())
qmax = max(previous_max, tensor.max())
```

典型 tag 包括：

```text
patch.conv
tokens
layers.0.norm1
layers.0.q
layers.0.k
layers.0.v
layers.0.attn_scores
layers.0.attn_probs
layers.0.attn
layers.0.proj
layers.0.resid1
layers.0.norm2
layers.0.fc1
layers.0.mlp_mid
layers.0.fc2
layers.0.resid2
final_norm
output.cls
```

tag 是双图之间的接口，不能使用自动生成的 PyTorch module 名或 TFDL symbol 名作为长期契约。

### 5.3 校准数据

`_load_calibration_samples` 支持常见图片和 `.npy`：

- 图片会 resize，并按 `rescale/mean/std` 转成原始 PyTorch 模型接收的 normalized NCHW。
- `.npy` 在当前校准入口中被视为已经预处理好的 NCHW。
- 没有提供可用数据时，会退化为随机正态输入。

随机输入只能用于打通流程，正式量化必须使用与实际业务分布一致的代表性数据。否则 range JSON 能生成，但量化精度没有保证。

## 6. TFDL 直接构图

### 6.1 权重和输入

所有 canonical float 权重先统一注册：

```python
ctx = TFContext("vit_op")
ctx.RegisterParamToContext(**weights)
```

`Op.Placeholder2` 将预处理嵌入输入节点。当前实现让 TFDL 接收原始 `[0,255]` NCHW，并配置：

```text
scale = rescale_factor / image_std
mean  = image_mean / rescale_factor
```

该配置对应：

```text
normalized = (raw * rescale_factor - image_mean) / image_std
```

数值对拍时必须区分：原始 Hugging Face/PyTorch 图接收 normalized 输入，TFDL executor 接收 raw 输入。

### 6.2 Block 构建

`_build_block_op` 的执行顺序为：

```text
LayerNorm
  → Q/K/V Conv1x1
  → reshape heads
  → 可选 RoPE
  → QK MatMul + scale
  → Softmax
  → AttentionWeight×V MatMul
  → output Conv1x1
  → residual add
  → LayerNorm
  → MLP/Gated MLP Conv1x1
  → residual add
```

DINOv3 的 RoPE 通过 `Op.Custom(..., "ApplyRope", ...)` 构建，因此 dump 和运行前必须加载包含该算子的 AddOn 动态库。自定义算子还必须分别验证 float 和量化执行路径。

当前 PyTorch 等效图把 `config.layer_norm_eps` 传给 `F.layer_norm`，而 TFDL 侧 `Op.LayerNorm2` 接口没有显式 epsilon 参数。接入 epsilon 不同的模型时，必须确认 SDK 算子的默认值与模型一致；如果不一致，需要扩展算子或使用自定义实现，不能仅靠最终输出误差碰运气。

### 6.3 建立逻辑 tag 到实际 symbol 的映射

TFDL 节点名称通常由 SDK 自动生成，例如 `TFDL_Convolution2_17`。构图时使用 `_mark` 保存映射：

```python
def _mark(symbol_map, tag, symbol):
    symbol_map[tag] = str(symbol)
    return symbol
```

一个简化的 `symbol_map.json` 如下：

```json
{
  "layers.0.q": "TFDL_Convolution2_2",
  "layers.0.attn_probs": "TFDL_Softmax_0",
  "layers.0.resid1": "TFDL_Add_1"
}
```

PyTorch 和 TFDL 中同一 tag 的 shape 可以不同。例如 PyTorch 的投影结果可能为 `[1,S,C]`，TFDL 的标记点为 `[1,C,H,W]`。只要二者只是 reshape/transpose 关系，元素集合相同，单一全局 min/max 就可以安全复用。若中间夹有计算、截断、拼接或只选取部分 token，就必须为新的语义边界定义新 tag。

## 7. range JSON 到 TFDL 量化

### 7.1 JSON 格式

PyTorch 采集结果示例：

```json
{
  "layers.0.q": {
    "min": -2.1834,
    "max": 2.4017
  },
  "layers.0.attn_probs": {
    "min": 0.000013,
    "max": 0.1842
  }
}
```

完成 TFDL 构图后，`annotate_minmax_json_with_symbol_map` 可以额外写入 `tfdl_name`，方便排查：

```json
{
  "layers.0.q": {
    "min": -2.1834,
    "max": 2.4017,
    "tfdl_name": "TFDL_Convolution2_2"
  }
}
```

加载 range JSON 时仍以逻辑 tag 为主键，`tfdl_name` 只是诊断信息。这样即使 SDK 自动 symbol 编号发生变化，只要拓扑语义没有变化，range 文件仍有机会复用。

### 7.2 注册顺序

TFDL 图构建并 `SetOutputs` 后，根据 symbol map 注册范围：

```python
for tag, actual_name in symbol_map.items():
    if tag in ranges:
        qmin, qmax = ranges[tag]
        assert ctx.AddInt8Config(actual_name, float(qmax), float(qmin))
```

注意 `AddInt8Config` 的参数顺序是 `(name, max, min)`，与 JSON 中常见的 `{min,max}` 顺序相反。

随后只运行一次 SDK 量化图转换：

```python
calib = TFCalibration(
    ctx,
    CalibrationMode.Naive,
    {"UseHardware": False, "FrugalMode": True},
)
calib.Quantize(
    {input_name: TFDataType.TFDL_UINT8},
    stopquanttensors=tuple(output_names),
    MergeConcate=False,
    Perchannel=True,
)
```

当前实现把模型输出列入 `stopquanttensors`，以便保留部署侧需要的输出边界。新增多输出模型时必须确认量化后所有输出仍然存在且顺序正确。

### 7.3 为什么能缩短量化时间

传统校准需要针对每个样本运行 TFDL 软件图，再从执行器收集中间范围。双图方案改为：

1. 在 PyTorch CUDA 中批次化或逐样本执行等效图。
2. GPU 上完成大部分 Transformer 运算。
3. 只把每个 tag 的标量 min/max 传回 CPU。
4. 将范围注册到 TFDL 图后，执行一次 SDK `Quantize` 和 dump。

因此主要节省的是**校准数据前向执行和范围收集时间**。TFDL SDK 的权重量化、节点改写和模型序列化仍然会执行。加速收益取决于模型大小、样本数量、GPU 性能和 TFDL 软件执行器速度。

## 8. 正确的开发与验证顺序

新增类似模型时建议使用以下顺序，避免在错误的等效图上耗时收集 range：

### 阶段 A：解析原始模型

1. 固定模型版本、输入尺寸、输出定义和预处理。
2. 列出每个 block 的真实算子顺序。
3. 明确 QKV 是 packed 还是分离，是否有 GQA/MQA、register token、LayerScale、RoPE、relative bias、drop path 或 gated MLP。
4. 将源权重映射到 canonical namespace，并对每个参数检查 shape。

### 阶段 B：先实现 PyTorch 等效图

1. 按计划中的 NPU layout 和 1x1 Conv 拓扑实现等效图。
2. 使用 `eval()` 和 `torch.no_grad()`，禁用 Dropout/drop path 等训练行为。
3. 用原始模型与等效图比较最终输出。
4. 如果模型包含 CLS 和完整 token 输出，两者都要比较。
5. 对复杂 block 增加逐层中间 tensor 对拍，先定位第一个发生偏差的位置。

只有原始模型与 PyTorch 等效图通过后，才说明“计划部署的拓扑”在数学上成立。

### 阶段 C：实现 TFDL 浮点图

1. 复用相同 canonical 权重。
2. 每个 PyTorch `_record(tag)` 在 TFDL 图中建立对应 `_mark(tag, symbol)`。
3. 使用 TFDL 软件 executor 比较原始模型和 TFDL 浮点输出。
4. 自定义算子先做独立 op-level 对拍，再接入完整模型。

### 阶段 D：快速收集范围并量化

1. 准备代表性校准集。
2. 在 PyTorch 等效图上收集 range JSON。
3. 检查 range 的有限性、覆盖率和异常极值。
4. 重新构建完全相同的 TFDL 图并注册范围。
5. 调用 SDK `Quantize` 生成 `.quant.fb`。
6. 单独验证量化模型的输出精度和真实 NPU 性能。

当前 `Vit.py` 已实现原始模型 ↔ PyTorch 等效图、原始模型 ↔ TFDL 浮点图的比较，但没有把“量化模型数值对拍”作为自动强制门槛。后续模型不能因为 `.quant.fb` 成功生成就视为量化正确。

## 9. 当前命令示例

### 9.1 一次完成对拍、范围收集和导出

```bash
python ConvertTools/python/example/Vit.py \
  --arch vit \
  --model-path /path/to/hf-vit \
  --image-size 224 \
  --calib-dir ./calibration_images \
  --num-calib 32 \
  --device cuda \
  --compare-reference \
  --compare-tfdl-fp \
  --compare-json /tmp/vit_compare.json \
  --dump-minmax-json /tmp/vit_quant.json \
  --dump-symbol-map /tmp/vit_symbol_map.json \
  --dump-fb /tmp/vit.fb \
  --dump-quant-fb /tmp/vit.quant.fb
```

`--dump-minmax-json` 生成的 `/tmp/vit_quant.json` 就是本文所说的 range JSON。

### 9.2 复用已有范围

```bash
python ConvertTools/python/example/Vit.py \
  --arch vit \
  --model-path /path/to/hf-vit \
  --image-size 224 \
  --range-json /tmp/vit_quant.json \
  --dump-quant-fb /tmp/vit.quant.fb
```

只有模型权重、输入 shape、预处理和双图拓扑完全一致时才能复用旧 range。任何 block、激活函数、融合策略或量化边界变化后都应重新收集。

DINOv2/DINOv3 使用相同主流程，入口分别为：

```text
ConvertTools/python/example/Dinov2.py
ConvertTools/python/example/Dinov3.py
```

DINOv3 还需要通过 `--addon-path` 提供实现 `ApplyRope` 的 `libTFDLAddOn.so`。

## 10. 为新 Transformer 复用该方案

建议把新模型实现拆成以下组件：

```text
ModelOpConfig
├── from_model_path()       # 配置与预处理解析
└── validate()              # 固定 shape 和结构约束

load_weights()
└── canonicalize_weights()  # 源命名 → 内部稳定命名

TorchModelOpGraph
├── forward()               # 与目标 TFDL 拓扑等效
└── _record(tag, tensor)    # 范围采集边界

build_model_tfdl_graph()
├── RegisterParamToContext
├── Op.* 直接构图
└── _mark(tag, symbol)      # tag → SDK symbol

compare_with_reference()
collect_minmax_json()
quantize_with_ranges()
```

对于 LLM 或多模态模型，还应额外处理：

- 动态序列长度和 KV cache 的输入输出。
- GQA/MQA 中 Q head 与 KV head 数量不同。
- causal/padding mask 与 fused attention 的语义。
- RMSNorm、RoPE、SiLU/SwiGLU 和自定义算子的量化路径。
- embedding/lm_head 权重共享。
- 大权重模型的内存峰值和可能生成的 `.weights.fb64` sidecar。
- 多输入、多输出在量化 dump 后是否完整保留。

不要机械复制 ViT 的固定 `[1,S,C]` 假设。应复用的是“双图、canonical 权重、稳定 tag、范围映射、分阶段对拍”的方法。

## 11. 常见错误和检查项

| 问题 | 后果 | 检查方法 |
| --- | --- | --- |
| PyTorch 等效图仍用 Linear，TFDL 用 Conv | layout 或融合差异不易暴露 | 两图都使用规范化后的 Conv1x1 权重 |
| 原始模型和 TFDL 输入都使用 normalized 数据 | TFDL Placeholder 再次预处理 | 明确 raw 与 normalized 两套输入 |
| packed QKV 拆分 axis 错误 | Q/K/V 全部错位 | 检查源权重为 `[3C,C]` 并沿 axis 0 拆分 |
| Softmax axis 错误 | shape 正常但 Attention 语义错误 | 明确 score 为 `[heads,S,S]`，沿最后一维归一化 |
| 忽略 LayerScale 或错误融合 bias | 每层误差累积 | 验证 `W'` 和 `b'` 都乘相同输出 scale |
| LayerNorm epsilon 或 GeLU 近似模式不一致 | 深层网络误差逐层放大 | 核对 SDK 默认值，并对 norm/activation 做 op-level 对拍 |
| tag 两侧不是同一语义 tensor | 错误 range 被注册到节点 | 逐 tag 比较 min/max，必要时比较完整 tensor |
| range 缺失但仍继续量化 | 部分节点使用非预期范围 | 量化前做 symbol-map/range coverage 检查 |
| range 中出现 NaN/Inf 或极端离群值 | scale 无效或精度严重下降 | 保存前验证 `isfinite(min/max)` 并统计分布 |
| 用随机输入作为正式校准集 | `.quant.fb` 可生成但精度不可控 | 使用真实业务代表数据 |
| 修改拓扑后复用旧 JSON | tag 名相同但数值边界已变化 | range 文件记录模型哈希、shape 和实现版本 |
| 自定义算子只有 float 路径 | 浮点正确、量化运行失败 | 分别执行 float/int8 op-level 测试 |
| 只验证最终 CLS 输出 | token 顺序或局部层错误被掩盖 | 同时比较完整 hidden tokens 和关键中间层 |
| `AddInt8Config` 顺序写反 | max/min 注册错误 | 始终使用 `(name, qmax, qmin)` |

当前 `Vit.py` 会映射已知 tag，但不会强制报告“TFDL symbol 有 tag 而 range 缺失”或“range 中存在无法映射的 tag”。实现新模型时建议把覆盖率检查作为量化前的硬门槛，并在报告中保存：

```text
range tag 总数
TFDL symbol-map tag 总数
成功映射数
缺失 range 的 tag
无法映射到 TFDL 的 tag
模型/权重标识、输入 shape、预处理和代码版本
```

## 12. 验收标准

| 阶段 | 必须满足的条件 |
| --- | --- |
| 权重规范化 | 所有参数存在、shape 正确、QKV/MLP/LayerScale 映射明确 |
| PyTorch 等效图 | 对原始模型的 CLS、完整 tokens 和关键中间层达到预期误差 |
| TFDL 浮点图 | 对原始模型数值一致，输出 shape/顺序正确 |
| range JSON | 使用代表性数据、值全部有限、tag 覆盖符合预期 |
| 量化导出 | `AddInt8Config` 全部成功，`.quant.fb` 和 sidecar 完整生成 |
| 量化验证 | 软件/NPU 输出精度满足任务指标，多输出完整保留 |
| 性能验证 | Linear 投影确实落到 1x1 Conv，没有不必要的 layout 往返 |

## 13. 当前实现文件索引

| 文件 | 作用 |
| --- | --- |
| [`ConvertTools/python/example/Vit.py`](../ConvertTools/python/example/Vit.py) | 通用 ViT/DINO 双图、权重映射、对拍、range 收集和量化 |
| [`ConvertTools/python/example/Dinov2.py`](../ConvertTools/python/example/Dinov2.py) | 以 `arch=dinov2` 调用共享实现 |
| [`ConvertTools/python/example/Dinov3.py`](../ConvertTools/python/example/Dinov3.py) | 以 `arch=dinov3` 调用共享实现 |
| [`ConvertTools/README`](../ConvertTools/README) | 面向使用者的转换命令说明 |
| [`Doc/Model_Conversion_Skill.md`](Model_Conversion_Skill.md) | Transformer 直接构图和自定义算子的通用背景资料 |

后续实现类似模型时，应优先复制这里的接口边界和验证顺序，而不是直接复制某个 ViT block 的固定 shape。真正可复用的核心是：**规范权重只做一次、部署与校准双图同构、量化节点使用稳定 tag 对齐、先浮点对拍再收集范围、最后由 SDK 完成量化图转换。**

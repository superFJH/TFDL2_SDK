# TFDL2 Transformer 模型转化 SKILL

基于 DINOv3 ViT 模型转化实践总结的完整工作流与注意事项，适用于 ViT / LLM / 多模态等 Transformer 架构模型的 NPU 部署。

---

## 目录

1. [转化总流程](#1-转化总流程)
2. [模型分析与权重准备](#2-模型分析与权重准备)
3. [Linear → 1x1 Conv 转换 (核心)](#3-linear--1x1-conv-转换-核心)
4. [计算图构建](#4-计算图构建)
5. [自定义算子开发](#5-自定义算子开发)
6. [量化与导出](#6-量化与导出)
7. [踩坑清单](#7-踩坑清单)
8. [API 速查与模式对照](#8-api-速查与模式对照)

---

## 1. 转化总流程

```
HuggingFace / PyTorch 模型
    │
    ├── 1. 分析架构: 哪些层, 哪些算子, 哪些 TFDL 不支持
    │
    ├── 2. 导出权重: safetensors → numpy dict
    │
    ├── 3. 开发自定义算子 (AddonOps/)
    │       ├── TFDL 内置不支持的操作 (如 RoPE, RMSNorm)
    │       ├── 编写 .cpp, cmake && make
    │       └── Python 测试脚本验证正确性
    │
    ├── 4. 构建计算图 (Python)
    │       ├── RegisterParamToContext 注册权重
    │       ├── Linear → Conv1x1 (关键优化)
    │       ├── GetParamSymbol 获取 TFSymbol
    │       └── Op.* 搭建网络
    │
    ├── 5. 量化 (Calibration)
    │       ├── TFCalibration 采集量化参数
    │       ├── Quantize → uint8
    │       └── Dump 导出 .fb 文件
    │
    └── 6. NPU 部署
            ├── TFExecutor + UseHardware=True
            └── 或 C++ 端加载 .fb 推理
```

---

## 2. 模型分析与权重准备

### 2.1 权重格式

从 safetensors 加载:

```python
from safetensors.torch import load_file

state_dict = load_file("model.safetensors")
weights = {k: v.numpy().astype(np.float32) for k, v in state_dict.items()}
```

多文件加载 (大模型):

```python
index_file = path / "model.safetensors.index.json"
with open(index_file) as f:
    index = json.load(f)
state_dict = {}
for wf in set(index["weight_map"].values()):
    sd = load_file(str(path / wf))
    state_dict.update(sd)
```

### 2.2 权重形状转换 (Linear → Conv1x1)

**这是最关键的转换**: NPU 对 Conv 有硬件加速，MatMul 不一定有。因此所有 `nn.Linear` 的权重需要从 2D 转为 4D。

| PyTorch 层 | safetensors 权重形状 | NPU 权重形状 | 说明 |
|-----------|---------------------|-------------|------|
| `nn.Linear(in, out)` | `(out, in)` | `(out, in, 1, 1)` | 末尾加两个 1 |
| `nn.Conv2d(in, out, k)` | `(out, in, kH, kW)` | 不变 | 已是正确格式 |
| `nn.LayerNorm(d)` | `(d,)` | 不变 | scale/bias |

**转换方法 (两种):**

```python
# 方法 1: 在权重 dict 里预先 reshape
weights = {}
for name, tensor in state_dict.items():
    arr = tensor.numpy().astype(np.float32)
    # Linear 权重: 2D → 4D
    if arr.ndim == 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1], 1, 1)
    weights[name] = arr

# 方法 2: 注册权重后再 reshape (不推荐, 会在图构建时出错)
# 推荐方法 1: 在加载阶段就处理好
```

### 2.3 注册权重到 Context

```python
ctx = TFContext("my_model")
ctx.RegisterParamToContext(**weights)
ctx.RegisterParamToContext(
    rope_sin=sin_np,   # 预计算的常量也可注册
    rope_cos=cos_np,
)
```

> **注意**: `RegisterParamToContext` 的 key 如果包含点号 (如 `encoder.layer.0.norm1.weight`)，
> 必须通过 `**dict` 解包传入，不能作为关键字参数直接传。

---

## 3. Linear → 1x1 Conv 转换 (核心)

### 3.1 为什么要转

NPU 芯片 (TF16110/TF7000) 对 Convolution 有专用硬件加速，MatMul 通常是软件实现或效率较低。
1x1 Conv 数学上等价于 Linear，但能利用 NPU 的卷积加速单元。

### 3.2 转换模式

**Linear (MatMul) 版本:**
```python
# 输入: (B, S, hidden_size) — 序列格式
Q = Op.MatMul2(normed, wq, bq, transA=False, transB=True)
```

**Conv1x1 版本 (推荐):**
```python
# 输入需要先转为空间格式: (B, S, hs) → (B, hs, H, W)
normed_spatial = Op.Transpose(normed, (0, 2, 1))          # (B, hs, S)
normed_spatial = Op.Reshape(normed_spatial, (B, hs, H, W)) # (B, hs, H, W)

# 1x1 Conv 等价于 Linear: weight 形状 (out, in, 1, 1)
Q = Op.Convolution2(normed_spatial, wq, bq,
                    kernel=1, pad=0, stride=1, dilation=1,
                    outChannel=hs, group=1)
# 输出: (B, outChannel, H, W)
```

### 3.3 完整的 Transformer Block Conv 版本

数据流:
```
输入: (B, S, hs)                        ← 序列格式
  │
  ├─ Transpose(0,2,1) → (B, hs, S)
  ├─ Reshape(B, hs, H, W)               ← 转为空间格式
  │
  ├─ QKV Conv1x1 × 3                    ← NPU 加速
  ├─ Reshape + Transpose → (B, nh, S, hd)
  ├─ RoPE (自定义算子)
  ├─ Attention (MatMul, NPU 无专用加速, 但不可避免)
  ├─ Transpose + Reshape → (B, hs, H, W)
  ├─ Output Conv1x1                      ← NPU 加速
  │
  ├─ Reshape(B, hs, S) + Transpose → (B, S, hs)
  ├─ 残差 Add
  │
  ├─ LayerNorm
  ├─ Transpose + Reshape → (B, hs, H, W)
  ├─ MLP Conv1x1 × 2 或 3               ← NPU 加速
  ├─ Reshape + Transpose → (B, S, hs)
  ├─ 残差 Add
  │
  输出: (B, S, hs)                       ← 回到序列格式
```

关键代码骨架:
```python
def _build_block_conv(ctx, hidden, config, layer_id, rope_sin, rope_cos):
    hs = config.hidden_size
    nh = config.num_attention_heads
    hd = config.head_dim
    H, W = config.h_patch, config.w_patch

    # 获取权重 (注意: 权重已 reshape 为 4D)
    wq = ctx.GetParamSymbol(f"encoder.layer.{layer_id}.attention.query.weight")
    bq = ctx.GetParamSymbol(f"encoder.layer.{layer_id}.attention.query.bias")
    # ... 其他权重 ...

    # === Pre-Attention LayerNorm ===
    normed = Op.LayerNorm2(hidden, norm1_w, norm1_b, axis=-1)

    # === 序列 → 空间 ===
    normed = Op.Transpose(normed, (0, 2, 1))            # (B, hs, S)
    normed = Op.Reshape(normed, (0, hs, H, W))          # (B, hs, H, W)

    # === QKV 投影 (Conv1x1) ===
    Q = Op.Convolution2(normed, wq, bq, kernel=1, pad=0,
                        stride=1, dilation=1, outChannel=hs, group=1)
    K = Op.Convolution2(normed, wk, bk, kernel=1, pad=0,
                        stride=1, dilation=1, outChannel=hs, group=1)
    V = Op.Convolution2(normed, wv, bv, kernel=1, pad=0,
                        stride=1, dilation=1, outChannel=hs, group=1)

    # === Conv 输出 → 多头格式 ===
    # Q: (B, hs, H, W) → (B, nh, hd, HW) → (B, nh, HW, hd)
    Q = Op.Reshape(Q, (1, nh, hd, -1))
    Q = Op.Transpose(Q, (0, 1, 3, 2))                   # (B, nh, S, hd)
    # K, V 同理

    # === RoPE + Attention (与 MatMul 版本相同) ===
    Q, K = _apply_rope_pair(Q, K, rope_sin, rope_cos, layer_id)
    # ... Attention 计算 ...

    # === Attention 输出 → 空间 → Conv1x1 输出投影 ===
    attn_4d = Op.Reshape(attn_3d, (1, nh, -1, hd))
    attn_4d = Op.Transpose(attn_4d, (0, 1, 3, 2))       # (B, nh, hd, S)
    attn_flat = Op.Reshape(attn_4d, (1, hs, H, W))       # (B, hs, H, W)

    proj = Op.Convolution2(attn_flat, wo, bo, kernel=1, pad=0,
                           stride=1, dilation=1, outChannel=hs, group=1)

    # === 空间 → 序列 (恢复残差连接格式) ===
    proj = Op.Reshape(proj, (1, hs, -1))                 # (B, hs, S)
    proj = Op.Transpose(proj, (0, 2, 1))                 # (B, S, hs)

    hidden = Op.Add(hidden, proj)                         # 残差

    # === MLP (同理: 序列→空间→Conv1x1→空间→序列) ===
    normed2 = Op.LayerNorm2(hidden, norm2_w, norm2_b, axis=-1)
    normed2 = Op.Transpose(normed2, (0, 2, 1))
    normed2 = Op.Reshape(normed2, (0, hs, H, W))
    # ... Conv1x1 MLP ...
    mlp_out = Op.Convolution2(mid, w_down, b_down, kernel=1, ...)
    mlp_out = Op.Reshape(mlp_out, (1, hs, -1))
    mlp_out = Op.Transpose(mlp_out, (0, 2, 1))

    output = Op.Add(hidden, mlp_out)
    return output
```

### 3.4 需要保留 MatMul 的场景

以下操作**无法**用 Conv1x1 替代，必须用 MatMul:

| 操作 | 原因 |
|------|------|
| Attention: Q @ K^T | 矩阵乘法，非 per-pixel 投影 |
| Attention: probs @ V | 同上 |
| Variable-length 序列 | Conv 需要固定的空间维度 H×W |

---

## 4. 计算图构建

### 4.1 核心模式

```python
ctx = TFContext("model_name")

# 注册权重
ctx.RegisterParamToContext(**weights)
ctx.RegisterParamToContext(rope_sin=sin_np, rope_cos=cos_np)

# 构建图
with ctx:
    # 获取权重符号
    w = ctx.GetParamSymbol("encoder.layer.0.attention.query.weight")
    rope_sin = ctx.GetParamSymbol("rope_sin")

    # 输入
    x = Op.Placeholder2(ctx, shape=(1, 3, 224, 224), outDatatype=TFDataType.TFDL_FLOAT)

    # 搭建网络
    out = Op.Convolution2(x, w, b, ...)
    out = Op.LayerNorm2(out, scale, bias, axis=-1)
    out = Op.Custom((Q, K, rope_sin, rope_cos), ...)  # 自定义算子

    # 在 with 块内记录输出节点名
    output_name = str(out)

# ⚠️ SetOutputs 必须在 with ctx 外面！
ctx.SetOutputs([output_name])

# 创建 Executor
executor = TFExecutor(context=ctx, config={"UseHardware": False, "FrugalMode": True})
```

### 4.2 关键规则

| 规则 | 说明 |
|------|------|
| `SetOutputs` 必须在 `with ctx` 外 | `with ctx` 的 `__exit__` 会 finalize 图结构，之后再 `SetOutputs` 才生效 |
| `str(TFSymbol)` 在 `with` 内获取 | `with` 退出后自动生成节点名，但要在 with 内用 `str()` 捕获 |
| `Op.Custom` 多输出返回 `list` | `results = Op.Custom(...)` 返回 `list[TFSymbol]`，单输出返回 `TFSymbol` |
| `Crop2` 参数必须是 TFSymbol | 不能传 int，改用 `Gather2(input, index, axis)` |
| `Reshape` 中 `-1` 可用于自动推导 | `Op.Reshape(x, (1, nh, -1, hd))` 中 `-1` 自动计算 |

---

## 5. 自定义算子开发

### 5.1 何时需要自定义算子

TFDL 内置算子清单 (无需自定义):
- Convolution2, MatMul/MatMul2, LayerNorm2
- GeLU, ReLU, Swish, Sigmoid, Softmax
- Add, Mul, Reshape, Transpose, Concat, Gather2

需要自定义算子的典型场景:
- **RoPE** (旋转位置编码): ApplyRope / ApplyRope3D
- **RMSNorm**: 比 LayerNorm 少均值计算
- **Gated MLP**: SquaredReLUGate / SwiGLU
- **KV-Cache Attention**: SDPAttentionKVCache
- **特殊激活函数**: 非标准激活

### 5.2 自定义算子模板

```cpp
// AddonOps/MyOp.cpp — 放入 AddonOps/ 目录，CMake 自动收录
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <cstring>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

using namespace TFDL_CAPI;
namespace TFDLOP {
    namespace MyOp {
        struct MyOpParam {
            float eps = 1e-5f;
            bool useFp16 = false;
        };

        void Prepare(TFContext tfContext, TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node), err);
            MyOpParam *p = new MyOpParam();
            if (err.empty()) {
                p->eps = (float)param["eps"].number_value();
                p->useFp16 = param["useFp16"].bool_value();
            }
            FreeNodeCustomParam(node, [](void *cp) { delete (MyOpParam *)cp; });
            NewNodeCustomParam(node, [&p]() -> void * { return p; });
        }

        void Reshape(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            // ⚠️ GetParam 回退: 支持 Placeholder 和 RegisterParamToContext 两种输入
            auto inputData = GetTensorByName(tfContext, info.InputNames[0]);
            if (!inputData.IsValid()) {
                inputData = GetParam(tfContext, info.InputNames[0]);
            }
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);
            ReSizeTensor(outputData, GetTensorShape(inputData));
            SetTensorType(outputData, GetTensorType(inputData));
        }

        void Eval(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            auto param = (MyOpParam *)GetNodeCustomParam(node);

            // 同样需要 GetParam 回退
            auto inputData = GetTensorByName(tfContext, info.InputNames[0]);
            if (!inputData.IsValid()) inputData = GetParam(tfContext, info.InputNames[0]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);

            if (GetTensorType(inputData) == TFCAPI_FLOAT) {
                // float 路径
                const float *in = (const float *)GetTensordata(inputData);
                float *out = (float *)GetTensordata(outputData);
                // ... NEON / 标量计算 ...
            } else if (GetTensorType(inputData) == TFCAPI_UINT8) {
                // uint8 路径 (NPU 量化推理): 反量化→计算→量化
                auto quant = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
                int total = 1;
                for (auto d : GetTensorShape(inputData)) total *= d;
                float *fData = new float[total];
                DeQuantizeTensorData(fData, (uint8_t*)GetTensordata(inputData), total, quant);
                // ... float 计算 ...
                auto outQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);
                QuantizeTensorData((uint8_t*)GetTensordata(outputData), fData, total, outQuant);
                delete[] fData;
            }
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *cp) { delete (MyOpParam *)cp; });
        }
    }
    RegistOp(MyOp).Set(MyOp::Prepare, MyOp::Reshape, MyOp::Eval, MyOp::Free);
}
```

### 5.3 自定义算子优化技巧

**NEON SIMD 加速层级:**

| 层级 | 技术 | 每迭代处理元素 | 条件 |
|------|------|--------------|------|
| L1 | 标量 | 1 | 无 |
| L2 | float32 NEON | 4 | `#ifdef __aarch64__` |
| L3 | FP16 NEON | 8 | `#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC`` |
| L4 | OpenMP 多线程 | × CPU 核心数 | `#pragma omp parallel for` |

**OpenMP 模式:**
```cpp
// b/h 循环独立，collapse(2) 合并为一个并行域
#pragma omp parallel for collapse(2) schedule(static)
for (int b = 0; b < B; b++) {
    for (int h = 0; h < numHeads; h++) {
        // 每个线程写独立输出行，无竞争
    }
}
```

**CMakeLists.txt 配置:**
```cmake
# FP16 NEON
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(TFDLAddOn PRIVATE -march=armv8.2-a+fp16)
endif()

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(TFDLAddOn PUBLIC OpenMP::OpenMP_CXX)
endif()
```

### 5.4 自定义算子测试

```python
# 标准测试模式: PyTorch 参考 vs TFDL 输出
import numpy as np, torch, json
from TFDL2 import TFContext, TFExecutor, Op
from TFDL2.Common import TFDataType
from TFDL2.utils import LoadCustomOp

LoadCustomOp("path/to/libTFDLAddOn.so")

# 1. PyTorch 参考实现
def reference_fn(x):
    return expected_output

expected = reference_fn(torch.from_numpy(input_np)).numpy()

# 2. TFDL2 计算图
ctx = TFContext("test")
with ctx:
    x = Op.Placeholder2(ctx, shape=input_np.shape, outDatatype=TFDataType.TFDL_FLOAT)
    result = Op.Custom((x,), ("out",), "MyOp", json.dumps({"eps": 1e-5}))
    ctx.SetOutputs(["out"])  # ⚠️ 必须在 with 外
executor = TFExecutor(context=ctx, config={"UseHardware": False, "FrugalMode": True})
executor.GetInputs()[0].fromNumpy(input_np)
actual = executor()[0].toNumpy()

# 3. 对比
assert np.allclose(actual, expected, atol=1e-5)
```

---

## 6. 量化与导出

### 6.1 量化流程

```python
from TFDL2 import TFCalibration, CalibrationMode

# 1. 创建 Calibration 对象
calibration = TFCalibration(ctx, CalibrationMode.Naive)

# 2. 填入校准数据
inputs = calibration.GetInputs()
for data in inputs:
    if data.dtype in (TFDataType.TFDL_INT32, TFDataType.TFDL_INT64):
        example = np.random.randint(0, 10000, data.shape).astype(tfdtype2npdtype(data.dtype))
    else:
        example = np.random.random(data.shape).astype(tfdtype2npdtype(data.dtype))
    data.fromNumpy(example)

# 3. 运行校准
calibration()

# 4. 指定量化类型 (通常全用 uint8)
inputtype = {data.name: TFDataType.TFDL_UINT8 for data in inputs}
calibration.Quantize(inputtype, MergeConcate=False)

# 5. 导出
ctx.Dump("model.quant.fb")
```

### 6.2 类型转换辅助

```python
def tfdtype2npdtype(dtype):
    return {
        TFDataType.TFDL_FLOAT: np.float32,
        TFDataType.TFDL_INT32: np.int32,
        TFDataType.TFDL_INT64: np.int64,
        TFDataType.TFDL_UINT8: np.uint8,
        TFDataType.TFDL_FLOAT16: np.float16,
    }.get(dtype, np.float32)
```

---

## 7. 踩坑清单

### 7.1 `with ctx` 与 `SetOutputs` 的顺序 (最关键!)

```python
# ❌ 错误: SetOutputs 在 with 块内 — 无效！
with ctx:
    out = Op.Add(x, y)
    ctx.SetOutputs([str(out)])   # with 退出时会被覆盖

# ✅ 正确: 先在 with 内捕获名字，退出后再 SetOutputs
with ctx:
    out = Op.Add(x, y)
    output_name = str(out)       # 捕获节点名
ctx.SetOutputs([output_name])    # 在 with 外设置
```

**原因**: `with ctx` 的 `__exit__` 会 finalize 图结构，之后再调 `SetOutputs` 才能正确生效。

### 7.2 Custom Op 必须支持 GetParam 回退

当 Python 端通过 `GetParamSymbol` 传入参数时，Custom Op 内部收到的 InputName 对应 param 名而非 tensor 名:

```cpp
// ✅ 正确: 先尝试 tensor，失败回退 param
auto data = GetTensorByName(tfContext, info.InputNames[i]);
if (!data.IsValid()) {
    data = GetParam(tfContext, info.InputNames[i]);
}
```

### 7.3 权重形状必须与算子匹配

| 算子 | 期望权重形状 | 常见错误 |
|------|------------|---------|
| `Convolution2` | `(outCh, inCh, kH, kW)` | 传了 2D `(out, in)` → 报错 |
| `MatMul2` | `(in, out)` 或 `(out, in)` | 需注意 `transB` 参数 |
| `LayerNorm2` | `(dim,)` | shape 不匹配 dim |

**关键**: 转为 Conv1x1 后，Linear 权重必须从 `(out, in)` reshape 为 `(out, in, 1, 1)`。

### 7.4 Op.Custom 多输出返回 list

```python
# 单输出: 返回 TFSymbol
result = Op.Custom((x,), ("out",), "RMSNorm", "{}")
# result 可直接传给下一个 Op

# 多输出: 返回 list[TFSymbol]
results = Op.Custom((Q, K, sin, cos), ("q_out", "k_out"), "ApplyRope", "{}")
q_rope = results[0]
k_rope = results[1]
```

### 7.5 Reshape 中使用 -1

```python
# ✅ 正确: -1 自动推导该维度
Op.Reshape(x, (1, nh, -1, hd))

# ❌ 错误: 多个 -1
Op.Reshape(x, (1, -1, -1, hd))  # 无法推导
```

### 7.6 Gather2 vs Crop2

```python
# Crop2 要求所有参数为 TFSymbol，不能传 int
# ❌ Op.Crop2(x, 0, 10, -1, 1)   # TypeError

# ✅ 使用 Gather2 (接受 int/list 索引)
cls_token = Op.Gather2(hidden, 0, axis=1)
```

### 7.7 Conv1x1 的 transB 问题

使用 `Convolution2` 时**不需要** `transB` 参数。权重直接以 `(out, in, 1, 1)` 格式传入。
但使用 `MatMul2` 时需要 `transB=True` 来匹配 PyTorch 的 `(out, in)` 格式。

### 7.8 LayerScale 权重处理

LayerScale 的 `lambda1` 是 `(hidden_size,)` 的一维向量:
```python
# 通过 Mul 广播到每个位置
ls1 = ctx.GetParamSymbol(f"{pfx}.layer_scale1.lambda1")
proj = Op.Mul(proj, ls1)  # (B, S, hs) * (hs,) → 广播
```

---

## 8. API 速查与模式对照

### 8.1 PyTorch → TFDL 算子映射

| PyTorch | TFDL (MatMul 版) | TFDL (Conv 版, 推荐) |
|---------|-----------------|---------------------|
| `nn.Linear(in, out)` | `MatMul2(x, w, b, transB=True)` | `Convolution2(x_spatial, w4d, b, kernel=1, ...)` |
| `nn.Conv2d` | `Convolution2(x, w, b, ...)` | 同左 |
| `nn.LayerNorm(d)` | `LayerNorm2(x, scale, bias, axis=-1)` | 同左 |
| `nn.GELU()` | `GeLU(x)` | 同左 |
| `nn.SiLU()` | `Swish(x)` | 同左 |
| `torch.matmul` | `MatMul(A, B, transA, transB)` | 不适用 (Attention 必须用 MatMul) |
| `torch.cat` | `Concat(tuples, axis)` | 同左 |
| `torch.gather` | `Gather2(input, index, axis)` | 同左 |
| 自定义操作 | `Op.Custom(inputs, outputs, name, json)` | 同左 |

### 8.2 Context 管理 API

| API | 说明 |
|-----|------|
| `ctx.RegisterParamToContext(**weights)` | 注册权重 (key 含点号用 `**dict`) |
| `ctx.GetParamSymbol("name")` | 获取 TFSymbol，用于构图 |
| `ctx.GetParam("name")` | 获取 TFTensor，运行时读数据 |
| `ctx.SetOutputs(["name1", ...])` | **必须在 `with ctx` 外调用** |
| `ctx.Dump("path.fb")` | 导出 FlatBuffers |

### 8.3 Executor / Calibration

```python
# CPU 仿真推理
executor = TFExecutor(context=ctx, config={"UseHardware": False, "FrugalMode": True})
executor.GetInputs()[0].fromNumpy(data)
outputs = executor()
result = outputs[0].toNumpy()

# 量化导出
calibration = TFCalibration(ctx, CalibrationMode.Naive)
# ... 填入校准数据 ...
calibration.Quantize({name: TFDataType.TFDL_UINT8 for name in ...}, MergeConcate=False)
ctx.Dump("model.quant.fb")
```

---

## 附录: Conv 版 vs MatMul 版的 reshape 开销

Conv 版需要额外的 Transpose + Reshape 在序列和空间格式之间转换:

```
序列格式: (B, S, hidden_size)    — LayerNorm, Add, 自定义算子使用
    ↕ Transpose + Reshape
空间格式: (B, hidden_size, H, W) — Conv1x1 使用
```

这些操作在 TFDL2 中是零拷贝的 (仅改变 shape 描述，不移动数据)，所以没有额外性能开销。
在 H×W 能整除 S (即 S = H×W) 的前提下，转换是安全的。

**当序列长度不能分解为 H×W 时** (如 LLM 自回归解码, S=1):
- 只能用 MatMul 版本，不能用 Conv 版本
- 或者人为 pad 到合适的空间维度 (不推荐，浪费算力)

# TFDL2 SDK 开发实践指南

本指南总结基于 TFDL2 SDK 进行模型部署的完整工作流，涵盖自定义算子开发、Python 测试脚本编写、以及使用 RegisterParamToContext 构建模型计算图的全流程。

基于 DINOv3 ViT 模型部署实践总结。

---

## 目录

1. [整体工作流](#1-整体工作流)
2. [自定义算子开发 (AddonOps)](#2-自定义算子开发-addonops)
3. [Python 测试脚本编写](#3-python-测试脚本编写)
4. [模型计算图构建 (RegisterParamToContext 模式)](#4-模型计算图构建-registerparamtocontext-模式)
5. [关键踩坑记录](#5-关键踩坑记录)
6. [常用 TFDL Python API 速查](#6-常用-tfdl-python-api-速查)

---

## 1. 整体工作流

```
目标模型 (HuggingFace / PyTorch)
    │
    ├── 1. 分析模型架构，识别需要的算子
    │
    ├── 2. TFDL 内置算子能覆盖的？
    │       ├── Convolution2, MatMul2, LayerNorm2, GeLU, Softmax, ...
    │       └── 直接在 Python 图中调用 Op.*
    │
    ├── 3. 内置算子不够的？→ 开发自定义算子 (AddonOps/)
    │       ├── 编写 .cpp 文件
    │       ├── cmake && make → libTFDLAddOn.so
    │       └── Python 中 LoadCustomOp 加载
    │
    ├── 4. 编写 Python 测试脚本验证自定义算子正确性
    │       └── PyTorch 参考实现 vs TFDL2 输出，np.allclose 对比
    │
    └── 5. 构建完整模型计算图
            ├── 从 safetensors 加载权重
            ├── RegisterParamToContext 注册权重
            ├── GetParamSymbol 获取 TFSymbol
            └── Op.* 系列函数搭建网络
```

---

## 2. 自定义算子开发 (AddonOps)

### 2.1 文件结构

每个自定义算子放在 `AddonOps/` 目录下的独立 `.cpp` 文件中，CMakeLists.txt 中的 `aux_source_directory` 会自动收录新文件，**无需修改构建配置**。

```cpp
// AddonOps/MyOp.cpp
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <cstring>
#include <cassert>

using namespace TFDL_CAPI;

namespace TFDLOP {
    namespace MyOp {

        // ======== 1. 参数结构体 ========
        struct MyOpParam {
            float eps = 1e-5f;
            // 从 JSON 配置解析的参数
        };

        // ======== 2. Prepare: 解析 JSON 配置 ========
        void Prepare(TFContext tfContext, TFNode node) {
            // 解析 JSON 配置字符串
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node), err);
            if (!err.empty()) {
                printf("[MyOp] JSON parse error: %s\n", err.c_str());
            }

            auto *p = new MyOpParam();
            p->eps = (float)param["eps"].number_value();

            // 注册参数生命周期管理
            FreeNodeCustomParam(node, [](void *cp) { delete (MyOpParam *)cp; });
            NewNodeCustomParam(node, [&p]() -> void * { return p; });
        }

        // ======== 3. Reshape: 计算输出形状 ========
        void Reshape(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);

            // ⚠️ 重要：获取输入数据时，需同时支持 Placeholder 和 Param 两种来源
            // 方式一：网络中间层/输入输出 → GetTensorByName
            // 方式二：RegisterParamToContext 注册的参数 → GetParam
            // 推荐写法：先尝试 GetTensorByName，失败回退 GetParam
            auto inputData = GetTensorByName(tfContext, info.InputNames[0]);
            if (!inputData.IsValid()) {
                inputData = GetParam(tfContext, info.InputNames[0]);
            }

            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);

            // 设置输出形状和类型
            ReSizeTensor(outputData, GetTensorShape(inputData));
            SetTensorType(outputData, GetTensorType(inputData));
        }

        // ======== 4. Eval: 运行时执行 ========
        void Eval(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);

            // 同样需要 GetTensorByName + GetParam 回退
            auto inputData = GetTensorByName(tfContext, info.InputNames[0]);
            if (!inputData.IsValid()) {
                inputData = GetParam(tfContext, info.InputNames[0]);
            }
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);

            auto param = (MyOpParam *)GetNodeCustomParam(node);
            auto shape = GetTensorShape(inputData);

            if (GetTensorType(inputData) == TFCAPI_FLOAT) {
                // float 路径
                const float *in = (const float *)GetTensordata(inputData);
                float *out = (float *)GetTensordata(outputData);
                // ... 执行计算 ...

            } else if (GetTensorType(inputData) == TFCAPI_UINT8) {
                // uint8 路径: 反量化 → float 计算 → 重新量化
                int totalElements = 1;
                for (auto d : shape) totalElements *= d;
                auto quant = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
                float *floatData = new float[totalElements];
                DeQuantizeTensorData(floatData, (uint8_t *)GetTensordata(inputData),
                                     totalElements, quant);
                // ... float 计算 ...
                auto outQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);
                QuantizeTensorData((uint8_t *)GetTensordata(outputData), floatData,
                                   totalElements, outQuant);
                delete[] floatData;
            }
        }

        // ======== 5. Free: 清理参数 ========
        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *cp) { delete (MyOpParam *)cp; });
        }
    }

    // ======== 6. 注册算子 ========
    RegistOp(MyOp)
    .Set(MyOp::Prepare, MyOp::Reshape, MyOp::Eval, MyOp::Free);
}
```

### 2.2 构建命令

```bash
cd AddonOps && mkdir -p build && cd build && cmake .. && make -j$(nproc)
# 输出: libTFDLAddOn.so
```

### 2.3 关键注意事项

| 要点 | 说明 |
|------|------|
| **GetTensorByName vs GetParam** | 网络中间层/输入输出用 `GetTensorByName`；RegisterParamToContext 注册的参数用 `GetParam`。**推荐先尝试前者，失败回退后者** |
| **Eval 必须同时处理 float 和 uint8** | NPU 推理使用 uint8 量化，CPU 仿真使用 float |
| **多输出算子** | `info.OutputNames` 可包含多个名称，对应多个输出 |
| **JSON 配置** | 通过 `GetNodeCustomJsonStr(node)` 获取 Python 端传入的 JSON 字符串 |
| **参数生命周期** | 必须用 `FreeNodeCustomParam` + `NewNodeCustomParam` 管理，防止内存泄漏 |

---

## 3. Python 测试脚本编写

### 3.1 标准测试模式

```python
import numpy as np
import json
from TFDL2 import TFContext, TFExecutor, Op
from TFDL2.Common import TFDataType
from TFDL2.utils import LoadCustomOp

# 1. 加载自定义算子
LoadCustomOp("path/to/libTFDLAddOn.so")

# 2. 准备测试数据
input_np = np.random.randn(1, 4, 8, 64).astype(np.float32)

# 3. PyTorch 参考实现
def reference_fn(x):
    # ... 用 PyTorch/numpy 实现相同的计算逻辑 ...
    return expected_output

expected = reference_fn(torch.from_numpy(input_np)).numpy()

# 4. 构建 TFDL2 计算图
ctx = TFContext("test_myop")
with ctx:
    x = Op.Placeholder2(ctx, shape=input_np.shape, outDatatype=TFDataType.TFDL_FLOAT)
    result = Op.Custom(
        (x,),                          # 输入 (tuple)
        ("output",),                   # 输出名称 (tuple)
        "MyOp",                        # 算子名称 (与 RegistOp 匹配)
        json.dumps({"eps": 1e-5})      # JSON 配置
    )
    # result 是 TFSymbol (单输出) 或 list[TFSymbol] (多输出)

# ⚠️ SetOutputs 在 with ctx 外面调用
ctx.SetOutputs(["output"])

# 5. 创建 Executor 并执行
executor = TFExecutor(context=ctx, config={"UseHardware": False, "FrugalMode": True})
inputs = executor.GetInputs()
inputs[0].fromNumpy(input_np)
outputs = executor()
actual = outputs[0].toNumpy()

# 6. 对比验证
assert np.allclose(actual, expected, atol=1e-5), f"max diff: {np.max(np.abs(actual - expected))}"
print("PASSED!")
```

### 3.2 多输出自定义算子测试

```python
# Op.Custom 对多输出算子返回 list[TFSymbol]
results = Op.Custom(
    (Q, K, sin, cos),
    ("q_out", "k_out"),        # 2 个输出
    "ApplyRope",
    "{}"
)
q_out = results[0]   # 第一个输出的 TFSymbol
k_out = results[1]   # 第二个输出的 TFSymbol
```

### 3.3 端到端多阶段测试

当一个算子的输出需要作为另一个算子的输入时，可以拆分为多个 Executor：

```python
# 阶段 1: 生成 sin/cos
ctx1 = TFContext("stage1")
with ctx1:
    # ...
    Op.Custom(..., ("sin_out", "cos_out"), "RopePositionEmbedding", ...)
ctx1.SetOutputs(["sin_out", "cos_out"])
exec1 = TFExecutor(context=ctx1, config=config)
# ... 执行 ...
sin_np = exec1()[0].toNumpy()
cos_np = exec1()[1].toNumpy()

# 阶段 2: 使用 sin/cos
ctx2 = TFContext("stage2")
with ctx2:
    sin_ph = Op.Placeholder2(ctx2, shape=sin_np.shape, ...)
    cos_ph = Op.Placeholder2(ctx2, shape=cos_np.shape, ...)
    # ...
```

---

## 4. 模型计算图构建 (RegisterParamToContext 模式)

### 4.1 完整流程

```python
from safetensors.torch import load_file

# Step 1: 加载 safetensors 权重
state_dict = load_file("model.safetensors")
weights = {k: v.numpy().astype(np.float32) for k, v in state_dict.items()}

# Step 2: 创建 Context
ctx = TFContext("my_model")

# Step 3: 注册所有权重 (key 含点号，通过 **dict 解包)
ctx.RegisterParamToContext(**weights)

# Step 4: 预计算常量 (如 RoPE sin/cos) 也可注册
sin_np, cos_np = precompute_rope(...)
ctx.RegisterParamToContext(rope_sin=sin_np, rope_cos=cos_np)

# Step 5: 构建计算图
with ctx:
    # 获取权重 TFSymbol
    w1 = ctx.GetParamSymbol("encoder.layer.0.attention.query.weight")
    b1 = ctx.GetParamSymbol("encoder.layer.0.attention.query.bias")
    rope_sin = ctx.GetParamSymbol("rope_sin")

    # 网络输入
    x = Op.Placeholder2(ctx, shape=(1, 3, 224, 224), outDatatype=TFDataType.TFDL_FLOAT)

    # 使用内置算子 (权重通过 TFSymbol 传入)
    out = Op.Convolution2(x, w_conv, b_conv, kernel=(3,3), ...)
    out = Op.LayerNorm2(out, w_ln, b_ln, axis=-1)
    out = Op.MatMul2(out, w1, b1, transA=False, transB=True)

    # 使用自定义算子 (GetParamSymbol 传入的 TFSymbol 也能工作)
    result = Op.Custom((Q, K, rope_sin, rope_cos), ...)

    # ⚠️ 在 with 块内记录输出节点名
    output_name = str(result)

# ⚠️ 关键: SetOutputs 必须在 with ctx 外面调用！
ctx.SetOutputs([output_name])

# Step 6: 创建 Executor
executor = TFExecutor(context=ctx, config={"UseHardware": False, "FrugalMode": True})

# Step 7: 推理
executor.GetInputs()[0].fromNumpy(image_np)
output = executor()[0].toNumpy()
```

### 4.2 GetParam vs GetParamSymbol

| API | 返回类型 | 用途 |
|-----|---------|------|
| `ctx.GetParam("name")` | `TFTensor` | 运行时读取参数数据，可调用 `.toNumpy()` |
| `ctx.GetParamSymbol("name")` | `TFSymbol` | 获取参数的图节点符号，用于 `Op.*` 构图 |

### 4.3 权重格式约定

| 来源算子 | safetensors 权重形状 | TFDL 调用方式 |
|---------|---------------------|--------------|
| `nn.Linear(in, out)` | `(out, in)` | `Op.MatMul2(x, w, b, transB=True)` |
| `nn.Conv2d(in, out, k)` | `(out, in, kH, kW)` | `Op.Convolution2(x, w, b, ...)` |
| `nn.LayerNorm(d)` | scale `(d,)`, bias `(d,)` | `Op.LayerNorm2(x, scale, bias, axis=-1)` |

**注意**: `transB=True` 使得 `x @ w.T`，匹配 PyTorch Linear 的语义。

---

## 5. 关键踩坑记录

### 5.1 `ctx.SetOutputs` 必须在 `with ctx` 之外

```python
# ❌ 错误: SetOutputs 在 with 块内
with ctx:
    out = Op.MatMul(...)
    ctx.SetOutputs([str(out)])   # 无效！with 退出时会覆盖

# ✅ 正确: SetOutputs 在 with 块外
with ctx:
    out = Op.MatMul(...)
    output_name = str(out)       # 在 with 内获取节点名

ctx.SetOutputs([output_name])    # 在 with 外设置输出
```

**原因**: `with ctx` 的 `__exit__` 会自动 finalize 图结构，之后再调用 `SetOutputs` 才能正确生效。

### 5.2 Custom Op 内部需支持 GetParam 回退

当 Python 端通过 `GetParamSymbol` 传入参数时，Custom Op 内部收到的 InputName 对应的是 param 名而非 tensor 名。因此 C++ 端需要：

```cpp
// ✅ 正确写法: 先尝试 tensor，失败回退 param
auto data = GetTensorByName(tfContext, info.InputNames[i]);
if (!data.IsValid()) {
    data = GetParam(tfContext, info.InputNames[i]);
}
```

这样同一个 Custom Op 既支持 `Placeholder2` 输入，也支持 `GetParamSymbol` 输入。

### 5.3 Op.Custom 多输出返回 list

```python
# 单输出: 返回 TFSymbol
result = Op.Custom((x,), ("out",), "RMSNorm", "{}")
# result 是 TFSymbol，可直接传入下一个 Op

# 多输出: 返回 list[TFSymbol]
results = Op.Custom((Q, K, sin, cos), ("q_out", "k_out"), "ApplyRope", "{}")
q_rope = results[0]
k_rope = results[1]
```

### 5.4 Crop2 等算子的参数必须是 TFSymbol

```python
# ❌ 错误: Crop2 要求所有参数为 TFSymbol
Op.Crop2(x, 0, 10, -1, 1)   # TypeError!

# ✅ 替代方案: 使用 Gather2 (接受 int 参数)
cls = Op.Gather2(hidden, 0, axis=1)  # 取 axis=1 位置 0
```

当需要从张量中提取特定位置的元素时，`Gather2` 接受 `int/list/tuple` 索引，比 `Crop2` 更方便。

### 5.5 获取自动生成的节点名

TFDL 内置算子的节点名是自动生成的（如 `TFDL_MATMUL_0`, `TFDL_ADD_1`），用 `str(TFSymbol)` 获取：

```python
out = Op.Add(x, y)
node_name = str(out)   # 例如 "TFDL_ADD_0"
ctx.SetOutputs([node_name])
```

### 5.6 RoPE 位置编码的预计算

对于固定图像尺寸的 ViT，RoPE sin/cos 可以在构图前用 numpy 预计算，然后通过 `RegisterParamToContext` 注册为参数，避免运行时重复计算。

---

## 6. 常用 TFDL Python API 速查

### 图构建 (需在 `with ctx:` 内)

| 操作 | API | 说明 |
|------|-----|------|
| 输入 | `Op.Placeholder2(ctx, shape, outDatatype)` | 定义输入占位符 |
| 卷积 | `Op.Convolution2(input, w, b, kernel, pad, stride, dilation, outCh, group)` | 带 weight/bias 的卷积 |
| 矩阵乘 | `Op.MatMul2(A, B, bias, transA, transB)` | 带偏置的矩阵乘 |
| 矩阵乘 | `Op.MatMul(A, B, transA, transB, hasBias)` | 无偏置或隐式偏置 |
| LayerNorm | `Op.LayerNorm2(input, scale, bias, axis)` | 带 scale/bias 的层归一化 |
| 激活 | `Op.GeLU(x)`, `Op.ReLU(x)`, `Op.Swish(x)`, `Op.Sigmoid(x)` | 常用激活函数 |
| Softmax | `Op.Softmax(x, axis)` | 指定轴的 softmax |
| 算术 | `Op.Add(a, b)`, `Op.Mul(a, b)` | 支持 TFSymbol + 标量 |
| Reshape | `Op.Reshape(x, shape)` | 形状变换，shape 是 tuple |
| Transpose | `Op.Transpose(x, dims)` | 维度转置 |
| Concat | `Op.Concat((a, b), axis)` | 沿指定轴拼接 |
| Gather | `Op.Gather2(input, indices, axis)` | 按索引取元素，indices 可以是 int |
| 自定义 | `Op.Custom(inputs, outputs, name, json_config)` | 调用 AddonOps 算子 |

### Context 管理

| 操作 | API | 说明 |
|------|-----|------|
| 注册权重 | `ctx.RegisterParamToContext(**weights)` | key 含点号可用 `**dict` 传入 |
| 获取参数符号 | `ctx.GetParamSymbol("name")` | 返回 TFSymbol，用于构图 |
| 获取参数数据 | `ctx.GetParam("name")` | 返回 TFTensor，可 `.toNumpy()` |
| 设置输出 | `ctx.SetOutputs(["name1", "name2"])` | **必须在 `with ctx` 之外调用** |
| 保存模型 | `ctx.Dump("path.fb")` | 导出为 FlatBuffers 格式 |

### Executor 推理

```python
executor = TFExecutor(context=ctx, config={"UseHardware": False, "FrugalMode": True})
inputs = executor.GetInputs()          # 获取输入 TFTensor 列表
inputs[0].fromNumpy(np_array)          # 设置输入数据
outputs = executor()                   # 执行推理，返回 TFTensor 列表
result = outputs[0].toNumpy()          # 获取输出 numpy 数组
```

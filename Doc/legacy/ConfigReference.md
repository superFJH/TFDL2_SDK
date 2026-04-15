### Config.json 与运行时配置参考

TFDL2 SDK通过JSON配置文件控制模型编译和运行时行为。配置分为两类：**config.json**（编译配置）和 **modify.json**（模型修改配置）。

---

## config.json — 编译配置

config.json用于`CompileExecutor()`，决定执行体的编译策略。配置直接影响推理性能和内存占用。

### 完整配置项

```json
{
    "UseHardware": true,
    "FrugalMode": true,
    "Eltmode": 2,
    "useCache": true,
    "cpuLimit": 1,
    "Core": [0],
    "optimize": {
        "DoAlign": false,
        "TryReverse": false,
        "HighAccuracy": true,
        "SplitDeconv": true,
        "SplitOp2MultiNPU": false,
        "MakeAlign": true,
        "MakeBroadCast": false,
        "MakeUnfold": true
    },
    "InputShape": [
        {"NodeName": "input.1", "Shape": [1, 3, 640, 640]}
    ]
}
```

### 配置项详细说明

#### UseHardware (bool)

是否使用NPU硬件加速。

- `true`：使用NPU硬件推理，获得最佳性能
- `false`：使用CPU软件推理，用于调试和结果校验

```json
"UseHardware": true
```

#### FrugalMode (bool)

内存节省模式。开启后执行体会在推理过程中复用中间张量的内存。

- `true`：内存复用，大幅减少内存占用，推荐在多路并发场景使用
- `false`：每个张量独立分配内存

```json
"FrugalMode": true
```

#### Core (array[int])

NPU核心绑定。指定执行体使用哪些NPU核心。

- NPU10T：只能使用单核心，每个执行体独占一个核心
- NPU40T：支持单执行体多核心

特殊值：
- `[-1]`：调度器分配绑定到大核（big core）
- `[-2]`：调度器分配绑定到小核（little core）
- `[0]`, `[1]`, `[2]`...：强制绑定到指定编号的核心，调度器不参与协调


```json
"Core": [-1]         // 使用大核
"Core": [-2]         // 使用小核
"Core": [0, 1, 2, 3] // 使用4个核心（NPU40T）
```

#### optimize (object)

模型编译优化选项。

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| DoAlign | bool | false | 内存对齐优化，加速NPU数据搬运 |
| TryReverse | bool | false | 尝试反转操作顺序以优化执行 |
| HighAccuracy | bool | true | 高精度模式，牺牲少量速度换取数值精度 |
| SplitDeconv | bool | true | 反卷积分割优化，将大核反卷积拆分 |
| SplitOp2MultiNPU | bool | false | 将操作拆分到多个NPU核心执行 |
| MakeAlign | bool | false | 生成对齐操作，确保内存对齐 |
| MakeBroadCast | bool | false | 生成广播操作，优化含有广播的网络 |
| MakeUnfold | bool | false | 生成展开操作，优化im2col过程 |

```json
"optimize": {
    "DoAlign": false,
    "TryReverse": false,
    "HighAccuracy": true,
    "SplitDeconv": true,
    "MakeAlign": true,
    "MakeUnfold": true
}
```

#### InputShape (array)

覆盖模型输入形状。用于动态输入模型或需要修改输入尺寸的场景。

```json
"InputShape": [
    {"NodeName": "input.1x", "Shape": [1, 3, 320, 320]}
]
```

当输入形状变化时，需要重新编译执行体：

```cpp
// C++ 示例：动态输入形状
std::string config = "{\"UseHardware\":true,\"FrugalMode\":true,\"InputShape\":[{\"NodeName\":\""
    + inputName + "\",\"Shape\":[1,3," + std::to_string(h) + "," + std::to_string(w) + "]}]}";
auto executor = CompileExecutor(context, true, config);
```

#### Eltmode (int)

元素操作的硬件执行码，用于判定硬件加速方式。

- `0`：不使用硬件加速
- `1`：标准模式
- `2`：标准模式，采用tflite的量化计算方法也是NPU中采用的算法

```json
"Eltmode": 2
```

#### useCache (bool)

是否启用缓存。

```json
"useCache": true
```

#### cpuLimit (int)

CPU软件推理时的线程数限制。

```json
"cpuLimit": 1
```

---

## modify.json — 模型修改配置

modify.json用于`ModifyTFContext()`，在运行时修改已加载的网络结构。适用于模型后处理、量化信息修正等场景。

### 完整配置结构

```json
{
    "AddOnPass": [],
    "DeleteLayer": [],
    "Layer": []
}
```

### 配置项详细说明

#### AddOnPass (array[string])

选择要运行的优化Pass。为空表示使用默认优化。

```json
"AddOnPass": ["MergeBatchnorm", "MergeBiasAdd"]
```

#### DeleteLayer (array[string])

要删除的节点名称列表。

```json
"DeleteLayer": ["Dropout_1", "Cast_2"]
```

#### Layer (array[object])

对指定层进行修改。支持以下几种修改模式：

##### 1. 修改量化信息

```json
{
    "layerName": "Softmax",
    "OutDataMin": [0],
    "OutDataMax": [1.0]
}
```

- `OutDataMin`：输出数据最小值，列表长度可以等于通道数（per-channel量化）
- `OutDataMax`：输出数据最大值

##### 2. 修改节点连接关系

```json
{
    "layerName": "Quantize0",
    "input": ["efficientnet-b0/model/blocks_0/batchnorm/add_1"],
    "output": ["Quantize0"]
}
```

- `input`：新的输入节点名称列表
- `output`：新的输出节点名称列表

##### 3. 修改输出数据类型

```json
{
    "layerName": "efficientnet-b0/model/head/dense/MatMul",
    "outputDataType": "TFDtypeFp32"
}
```

可选的 `outputDataType` 值：
- `"TFDtypeFp32"` — float32
- `"TFDtypeUint8"` — uint8
- `"TFDtypeInt32"` — int32

##### 4. 完整的单层描述（新增/替换层）

```json
{
    "input": [],
    "layerName": "data",
    "layerType": "SetInput",
    "output": ["data"],
    "outputDataType": "TFDtypeUint8",
    "ActivationType": "ReLU",
    "param": {
        "tfDataType": "TFDtypeUint8",
        "mean": [128, 128, 128],
        "scale": [0.008712500334, 0.008712500334, 0.008712500334],
        "shape": [1, 3, 300, 300]
    }
}
```

---

## 典型配置场景

### 场景1：单路NPU推理（默认配置）

```json
{
    "UseHardware": true,
    "FrugalMode": true,
    "Core": [-1],
    "optimize": {
        "HighAccuracy": true,
        "SplitDeconv": true
    }
}
```

### 场景2：NPU40T多核针对单一模型加速

```json
{
    "UseHardware": true,
    "FrugalMode": true,
    "Core": [0, 1, 2, 3],
    "useCache": false,#因为缓存是每个NPU独占，当多核加速时跨核心缓存一致性没法保证需要关闭
    "optimize": {
        "SplitOp2MultiNPU": true,
        "HighAccuracy": true
    }
}
```

### 场景3：CPU调试模式

```json
{
    "UseHardware": false,
    "FrugalMode": false,
    "cpuLimit": 1
}
```

### 场景4：动态输入形状（OCR等变尺寸任务）

```json
{
    "UseHardware": true,
    "FrugalMode": true,
    "Eltmode": 2,
    "useCache": true,
    "Core": [0],
    "optimize": {
        "MakeAlign": true,
        "MakeBroadCast": false,
        "MakeUnfold": true
    },
    "InputShape": [
        {"NodeName": "input", "Shape": [1, 3, 448, 960]}
    ]
}
```

注意：当输入形状变化时需要重新编译执行体。对于OCR检测等需要动态尺寸的场景，应在每次推理前重新调用`CompileExecutor`,也可以一次性编译常见的多种形状的执行体。

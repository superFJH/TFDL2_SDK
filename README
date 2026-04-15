# TFDL2 SDK

TFDL2 (Think-Force Deep Learning 2) 是由 [Think-Force](http://www.think-force.com/) 自主研发的轻量级量化推理框架，配合 Think-Force NPU 使用。内核完全由 C++ 编写，可轻松移植到嵌入式设备；对 x86、ARM 及 Think-Force 自研 ManyCore<sup>TM</sup> 架构进行了汇编级深度优化，保证神经网络的高速运行。

## 主要特性

- **丰富的算子支持** — 支持Convolution、FullyConnect等基础操作，以及[OctConv](https://arxiv.org/abs/1904.05049)、[SEBlock](https://arxiv.org/abs/1709.01507)、[ResidualBlock](https://arxiv.org/abs/1512.03385)等高级网络结构，同时支持C++自定义算子
- **灵活的数据格式** — 支持float和uint8等多种数据类型，提供不同级别的模型量化方案
- **全面的框架支持** — 支持 [Caffe](http://caffe.berkeleyvision.org/)、[TensorFlow](https://tensorflow.google.cn/)、[TensorFlowLite](https://tensorflow.google.cn/lite/)、[ONNX](https://onnx.ai/) 模型转换
- **多硬件平台** — 支持 TF16110 (NPU10T) 和 TF7000 (NPU40T) 等 NPU 芯片
- **双语言接口** — 提供 C/C++ API 和 Python SDK

## 目录结构

```
TFDL2_SDK/
├── include/            # 公共C/C++头文件
│   ├── TFDL2_C_API.h   #   核心API头文件
│   ├── TFDL2_Common.h  #   公共类型/枚举定义
│   ├── TFCV/           #   TFCV图像视频模块头文件
│   └── onnxruntime/    #   ONNX Runtime头文件
├── lib/                # 预编译共享库
│   ├── libTFDL2_C_API.so         # 完整运行时API
│   ├── libTFDL2_LITE_C_API.so    # 精简运行时API
│   ├── libNPU10T.so / libNPU40T.so  # NPU硬件驱动
│   ├── CV_NPU10T/      # TF16110芯片相关库
│   └── CV_NPU40T/      # TF7000芯片相关库
├── AddonOps/           # 自定义算子实现（构建生成libTFDLAddOn.so）
├── Python/             # Python SDK包（pybind11封装）
├── ConvertTools/       # 模型转换工具链
│   ├── exe/            #   二进制转换工具 TFConvertor
│   └── python/         #   Python转换工具 onnx2tfdl
├── Example/            # C++示例（分类、检测、分割、Benchmark等）
├── bin/                # 预编译可执行文件
└── Doc/legacy/         # 完整文档
```

## 支持的硬件平台

| 平台 | 芯片 | NPU核心 | 适用场景 |
|------|------|---------|---------|
| NPU10T | TF16110 | 4×2（4簇，每簇2核，簇间独立总线） | 边缘设备、加速卡 |
| NPU40T | TF7000 | 4大核+4小核 | 服务器加速卡、边缘服务器 |

### 双芯片NUMA配置

在双芯片 TF7000 平台（如7140服务器）上，NPU分布在不同的NUMA节点：

- **NUMA 0**：NPU 0 ~ NPU 7
- **NUMA 1**：NPU 8 ~ NPU 15

运行NPU相关程序时，需使用 `numactl` 绑定到对应的NUMA节点，以避免跨NUMA访问带来的性能损失：

```bash
# 使用NUMA 0上的NPU（NPU 0-7）
numactl --membind=0 --cpunodebind=0 ./your_program

# 使用NUMA 1上的NPU（NPU 8-15）
numactl --membind=1 --cpunodebind=1 ./your_program
```

## 快速开始

### 环境要求

- 操作系统：Linux（aarch64）
- Python版本：3.6及以上
- CMake 3.10及以上（编译自定义算子需要）

### 1. 安装 Python SDK

```bash
cd Python
pip install .
```

安装完成后即可使用 `TFDL2` Python包进行开发。

### 2. 模型转换与量化

#### 方式一：Python工具（推荐）

```bash
cd ConvertTools/python
python onnx2tfdl.py --onnx model.onnx --output model.fb
```

常用模型提供了一键转换示例，参见 `ConvertTools/python/example/` 目录，包含 YOLOv5、YOLOv8、YOLO11、YOLO26 等典型模型的转换脚本。

#### 方式二：二进制工具

```bash
cd ConvertTools/exe
# 模型转换（proto编号：1=Caffe, 2=TF, 3=TFLite, 4=ONNX, 5=TFDL重新量化）
./TFConvertor --proto 4 --model model.onnx --output model.fb
# INT8量化校准
./TFConvertor --proto 5 --model model.fb --quantize 1 --images calib_images/ --output model.quant.fb
```

### 3. 自定义算子编译

```bash
cd AddonOps
mkdir -p build && cd build
cmake .. && make -j$(nproc)
# 产物：libTFDLAddOn.so（运行时通过LoadCustomOp加载）
```

新增自定义算子只需在 `AddonOps/` 目录下添加 `.cpp` 文件，CMake 会自动发现并编译。

### 4. 运行推理

#### Python 示例

```python
from TFDL2 import TFContext, TFExecutor, Op
from TFDL2.Common import TFDataType
from TFDL2.utils import LoadCustomOp

# 加载自定义算子（如需要）
LoadCustomOp("libTFDLAddOn.so")

# 构建网络
ctx = TFContext("my_model")
with ctx:
    x = Op.Placeholder2(ctx, shape=[1, 3, 224, 224], outDatatype=TFDataType.TFDL_FLOAT)
    ctx.LoadProto("model.quant.fb")
    ctx.SetOutputs(ctx.GetOutputNames())

# 创建执行器并推理
executor = TFExecutor(context=ctx, config={"UseHardware": True, "FrugalMode": True})
inputs = executor.GetInputs()
inputs[0].fromNumpy(numpy_array)  # 填入输入数据
outputs = executor()               # 执行推理
result = outputs[0].toNumpy()      # 获取输出
```

#### C++ 示例

```bash
# 编译示例
cd Example/Classification
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# 运行
./Classification model.fb config.json image.jpg
```

#### Benchmark 性能测试

```bash
cd testbin
./Benchmark -1 1 100 0 runconfig.json
```

### 5. 硬件环境配置

设置 `LD_LIBRARY_PATH` 指向对应芯片的库目录：

```bash
# TF16110 (NPU10T)
export LD_LIBRARY_PATH=/path/to/TFDL2_SDK/lib/CV_NPU10T:$LD_LIBRARY_PATH

# TF7000 (NPU40T)
export LD_LIBRARY_PATH=/path/to/TFDL2_SDK/lib/CV_NPU40T:$LD_LIBRARY_PATH
```

或使用自动选择脚本：

```bash
cd bin
bash ZYNQ7020.sh model.quant.fb config.json  # 根据Ubuntu版本自动选择NPU10T/NPU40T
```

## 典型工作流程

```
训练模型 (PyTorch/TensorFlow)
        │
        ▼
导出 ONNX / Caffe / TFLite / TF
        │
        ▼
TFConvertor / Python转换工具
  ├── 模型格式转换
  ├── 图优化（合并BN、融合激活等）
  ├── INT8量化校准
  └── 保存为 .fb 文件
        │
        ▼
部署推理
  ├── C++ SDK: LoadProto → CompileExecutor → ForwardExecutorAlone
  ├── Python SDK: TFContext → TFExecutor → executor()
  └── 自定义算子: LoadCustomOp → Op.Custom
```

## 示例应用

| 目录 | 说明 |
|------|------|
| `Example/Classification/` | 图像分类（ResNet50） |
| `Example/yolov5/` | 目标检测（YOLOv5/YOLO11），含转换量化脚本 |
| `Example/CityScape/` | 语义分割（PPLite） |
| `Example/Benchmark/` | 性能基准测试工具 |
| `Example/rtspServer/` | RTSP视频流推理服务 |
| `Example/PythonDevelop/` | Python开发示例（Jupyter Notebook） |


## 文档

| 文档 | 说明 |
|------|------|
| [API参考手册](Doc/legacy/API.md) | C/C++ API完整参考：TFContext、TFExecutor、TFTensor、量化、自定义算子等 |
| [使用教程](Doc/legacy/Tutorial.md) | 快速上手：第一个TFDL2程序、模型转换工具、网络可视化、自定义层编写 |
| [算子参考](Doc/legacy/Operation.md) | 支持的全部算子/层详细说明：卷积、激活、归约、量化、高级层等 |
| [模型格式与量化](Doc/legacy/ModelAndQuantization.md) | FlatBuffers模型格式、INT8量化原理、校准算法（Naive/KLD/Mean/Coverage）、混合精度 |
| [性能基准测试](Doc/legacy/Benchmark.md) | 各类模型在NPU上的推理性能数据、Benchmark工具使用方法、性能优化建议 |
| [Python API手册](Doc/legacy/PythonAPI.md) | Python SDK使用：TFContext/TFExecutor/Op模块、模型转换、量化、推理示例 |
| [TFCV图像视频模块](Doc/legacy/TFCV.md) | TFCV视觉IO模块：图片解码、视频流读取、RTSP接入、颜色格式、多路并发 |
| [配置参考](Doc/legacy/ConfigReference.md) | config.json完整配置项说明、modify.json模型修改语法、典型配置场景 |
| [多线程编程指南](Doc/legacy/MultiThread.md) | 多执行体并发模式、NPU核心分配、权值共享、线程安全规则、性能调优 |

## 框架架构

TFDL2由上至下分为四层：

```
┌─────────────────────────────────┐
│         用户接口                  │  TFDL2_C_API / PyTFDL / TFDNN
├─────────────────────────────────┤
│         网络结构                  │  TFContext / Optimizer / Net / MemManager
├─────────────────────────────────┤
│         网络引擎                  │  TFScheduler / TFExecutor / TFCompiler
├─────────────────────────────────┤
│         硬件抽象                  │  TFHardware / TFDNN
└─────────────────────────────────┘
```

- **用户接口** — C/C++ API 适合高性能场景；Python接口方便快速开发
- **网络结构** — TFContext 保存网络上下文，Optimizer 提供图优化，Net 管理节点和连通性
- **网络引擎** — TFExecutor 编译执行网络，TFScheduler 调度多执行体，TFCompiler 针对硬件深度优化
- **硬件抽象** — 统一处理不同硬件差异，支持 GPU 和 NPU

## 支持的模型格式转换

| 源格式 | 转换类型编号 | 说明 |
|--------|------------|------|
| Caffe (prototxt + caffemodel) | 1 | 需要同时提供网络结构和权值文件 |
| TensorFlow (pb) | 2 | frozen graph格式 |
| TFLite (tflite) | 3 | TensorFlow Lite量化/浮点模型 |
| ONNX (onnx) | 4 | 推荐使用Python转换工具 |
| TFDL (.fb) | 5 | 对已有TFDL2模型重新量化 |

## 相关链接

- 官方网站：[http://www.think-force.com/](http://www.think-force.com/)

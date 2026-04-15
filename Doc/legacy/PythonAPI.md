### TFDL2 Python API 使用手册

TFDL2提供了完整的Python接口，通过pybind11封装C++核心API，支持模型构建、转换、量化和推理的全部功能。

#### 安装

```bash
# 进入SDK的Python目录
cd TFDL2_SDK/Python

# 安装TFDL2 Python包
pip install .
```

安装完成后即可在Python中导入：
```python
from TFDL2 import TFContext, TFExecutor, TFTensor, Op
from TFDL2.Common import TFDataType, CalibrationMode, DECODER_FLAGS
```

#### 核心模块结构

| 模块 | 说明 |
|------|------|
| `TFDL2.TFContext` | 网络上下文管理，模型加载与序列化 |
| `TFDL2.TFExecutor` | 编译后的执行体，负责推理 |
| `TFDL2.TFTensor` | 张量操作，与NumPy互转 |
| `TFDL2.Op` | 算子库，支持100+种神经网络操作 |
| `TFDL2.TFCalibration` | 模型校准与量化 |
| `TFDL2.Common` | 枚举类型定义 |
| `TFDL2.utils` | 自定义算子注册 |
| `TFDL2.Convertor` | 模型转换框架 |

#### 枚举类型

```python
from TFDL2.Common import TFDataType, CalibrationMode, DECODER_FLAGS

# 数据类型
TFDataType.FLOAT     # float32
TFDataType.UINT8     # uint8 (INT8量化后)
TFDataType.INT32     # int32
TFDataType.INT64     # int64
TFDataType.BFLOAT16  # bfloat16
TFDataType.UINT16    # uint16
TFDataType.FLOAT16   # float16

# 校准模式
CalibrationMode.Naive     # 最大最小值法
CalibrationMode.KLD       # KL散度法 (Entropy)
CalibrationMode.MEAN      # 均值法
CalibrationMode.COVERAGE  # 覆盖率法

# 图像颜色格式
DECODER_FLAGS.BGR   # BGR顺序
DECODER_FLAGS.RGB   # RGB顺序
DECODER_FLAGS.Gray  # 灰度
```

#### TFContext 使用

TFContext是网络图的Python封装，支持从文件加载和从零构建。

```python
from TFDL2 import TFContext, Op

# 方式1：从.fb文件加载已有模型
ctx = TFContext("model.fb")

# 方式2：使用with语句从零构建网络图
with TFContext("my_model") as ctx:
    # 创建输入占位符
    input = Op.Placeholder(ctx, "input", [1, 3, 224, 224])

    # 使用Op构建网络（返回TFSymbol，是惰性求值的）
    conv1 = Op.Convolution(ctx, input, outChannel=64,
                           kernel_w=7, kernel_h=7, stride_w=2, stride_h=2,
                           pad=[3,3,3,3], hasBias=False, group=1)
    bn1 = Op.Scale(ctx, conv1, hasBias=True)
    relu1 = Op.ReLU(ctx, bn1)
    pool1 = Op.MaxPooling(ctx, relu1, kernel_w=3, kernel_h=3,
                          stride_w=2, stride_h=2, SAME=True)
    # ... 继续构建网络

    # 设置输出节点
    ctx.SetOutputs(["output_name"])

# 序列化到文件
ctx.Dump("my_model.fb")

# 注册权值参数
import numpy as np
weight = np.random.randn(64, 3, 7, 7).astype(np.float32)
ctx.RegisterParamToContext(conv1_weight=weight)

# 获取权值参数
param = ctx.GetParam("conv1_weight")
data = param.toNumpy()  # 转为numpy数组
```

##### 上下文修改（JSON方式）

```python
# 修改量化信息
ctx.Modify('''
{
    "Layer": [
        {
            "layerName": "Softmax",
            "OutDataMin": [0],
            "OutDataMax": [1.0]
        }
    ]
}
''')

# 删除节点
ctx.Modify('''
{
    "DeleteLayer": ["node_to_remove"]
}
''')

# 注册量化范围
ctx.AddInt8Config("conv1_output", qmax=10.0, qmin=-10.0)
```

#### TFExecutor 使用

```python
from TFDL2 import TFContext, TFExecutor
import numpy as np

# 加载模型
ctx = TFContext("model.fb")

# 编译执行体
# Option 是默认配置字典，可以自定义
option = {
    "UseHardware": True,     # 使用NPU硬件
    "FrugalMode": True,      # 内存复用
    "Core": [],              # NPU核心绑定
    "optimize": {
        "DoAlign": False,
        "TryReverse": False,
        "HighAccuracy": True,
        "SplitDeconv": True
    },
    "InputShape": [
        {"NodeName": "input", "Shape": [1, 3, 224, 224]}
    ]
}
executor = TFExecutor(ctx, option)

# 获取输入输出张量信息
inputs = executor.GetInputs()
outputs = executor.GetOutputs()
for t in inputs:
    print(f"输入: {t.name}, 形状: {t.shape}, 类型: {t.dtype}")

# 写入输入数据
input_tensor = inputs[0]
img_data = np.random.randint(0, 255, (1, 3, 224, 224)).astype(np.uint8)
input_tensor.fromNumpy(img_data)

# 执行推理
results = executor()

# 读取输出
for output in results:
    print(f"输出: {output.name}, 形状: {output.shape}")
    data = output.toNumpy()  # 转为numpy数组
    print(f"数据类型: {data.dtype}, 范围: [{data.min()}, {data.max()}]")

# INT8量化模型的输出反量化
if output.dtype == TFDataType.UINT8:
    raw = output.toNumpy()  # uint8原始数据
    qscale = output.qscale
    qzeropoint = output.qzeropoint
    float_data = (raw.astype(np.float32) - qzeropoint) * qscale
```

#### TFTensor 属性与方法

```python
tensor = executor.GetInputs()[0]

# 属性
tensor.name          # 张量名称 (str)
tensor.shape         # 形状 (list[int])
tensor.dtype         # 数据类型 (TFDataType)
tensor.qscale        # 量化缩放因子 (float)
tensor.qzeropoint    # 量化零点 (int)
tensor.qmax          # 量化最大值 (float)
tensor.qmin          # 量化最小值 (float)

# 方法
data = tensor.toNumpy()              # 转为numpy数组
tensor.fromNumpy(np_array)           # 从numpy数组写入数据
```

#### Op 算子库

Op模块提供了100+种神经网络操作的Python接口。所有操作都是声明式的——它们向TFContext中添加算子节点，返回`TFSymbol`（代表张量名称），不会立即执行计算。

##### 算术运算

```python
# TFSymbol支持Python运算符重载
c = a + b      # Op.Add(ctx, a, b)
c = a - b      # Op.Sub(ctx, a, b)
c = a * b      # Op.Mul(ctx, a, b)
c = a / b      # Op.Div(ctx, a, b)
c = a < b      # Op.BoolLT(ctx, a, b)
c = a > b      # Op.BoolGT(ctx, a, b)
c = a == b     # Op.BoolEQ(ctx, a, b)

# 也支持标量运算
c = a * 2.0    # 标量乘法
c = a + 1.0    # 标量加法
```

##### 卷积与池化

```python
# 2D卷积
conv = Op.Convolution(ctx, input, outChannel=64,
                      kernel_w=3, kernel_h=3,
                      stride_w=1, stride_h=1,
                      pad=[1,1,1,1], group=1, hasBias=True)

# 反卷积
deconv = Op.DeConvolution(ctx, input, outChannel=256,
                          kernel_w=4, kernel_h=4,
                          stride_w=2, stride_h=2,
                          pad=[1,1,1,1], hasBias=True)

# 最大池化
pool = Op.MaxPooling(ctx, input, kernel_w=2, kernel_h=2,
                     stride_w=2, stride_h=2)

# 全局平均池化
gap = Op.GlobalAvePool(ctx, input)
```

##### 激活函数

```python
relu = Op.ReLU(ctx, input)
leaky = Op.LeakyReLU(ctx, input, negative_slope=0.1)
prelu = Op.PReLU(ctx, input)
relu_x = Op.ReLUX(ctx, input, threshold=6.0)
sigmoid = Op.Sigmoid(ctx, input)
tanh = Op.Tanh(ctx, input)
swish = Op.Swish(ctx, input)
hard_swish = Op.HardSwish(ctx, input)
gelu = Op.GeLU(ctx, input)
mish = Op.Mish(ctx, input)
elu = Op.ELU(ctx, input, alpha=1.0)
```

##### 形状操作

```python
reshaped = Op.Reshape(ctx, input, [1, -1])
squeezed = Op.Squeeze(ctx, input, axes=[2, 3])
flatten = Op.Flatten(ctx, input)
transposed = Op.Transpose(ctx, input, [0, 2, 3, 1])
expanded = Op.ExpandDims(ctx, input, axis=0)
broadcast = Op.BroadCast(ctx, input, target_shape=[1, 64, 1, 1])
```

##### 归约操作

```python
mean = Op.ReduceMean(ctx, input, axis=[2, 3])
sum = Op.ReduceSum(ctx, input, axis=[1])
max = Op.ReduceMax(ctx, input, axis=[1])
argmax = Op.ArgMax(ctx, input, axis=1)
```

##### 张量操作

```python
# 拼接
concat = Op.Concat(ctx, [tensor1, tensor2], axis=1)

# 切分
slices = Op.Split(ctx, input, split_sizes=[128, 128], axis=1)

# 切片
sliced = Op.Slice(ctx, input, starts=[0,0,0,0], ends=[-1,256,-1,-1])

# Gather
gathered = Op.Gather(ctx, input, indices, axis=1)

# 翻转
flipped = Op.Flip(ctx, input)

# Pad
padded = Op.Pad(ctx, input, paddings=[[0,0],[0,0],[1,1],[1,1]])

# 条件选择
result = Op.Where(ctx, condition, x, y)

# 类型转换
casted = Op.Cast(ctx, input, TFDataType.FLOAT)
```

##### 缩放操作

```python
# 双线性插值缩放
resized = Op.BilinearResize(ctx, input, target_height=224, target_width=224)
# 或者使用缩放因子
resized = Op.BilinearResizeByScale(ctx, input, scale=2.0)

# 最近邻缩放
resized = Op.NearestResize(ctx, input, target_height=448, target_width=448)
```

##### 全连接与矩阵运算

```python
# 全连接层（InnerProduct）
fc = Op.InnerProduct(ctx, input, outchannels=1000, hasBias=True)

# 矩阵乘法
matmul = Op.MatMul(ctx, a, b, transA=False, transB=False, hasBias=True)
```

##### 量化相关操作

```python
# 量化（float -> uint8）
quantized = Op.Quantize(ctx, input)

# 反量化（uint8 -> float）
dequantized = Op.DeQuantize(ctx, input)

# 重量化（uint8 -> uint8，不同量化参数间转换）
requantized = Op.Requantize(ctx, input)
```

##### 特殊操作

```python
# Softmax
softmax = Op.Softmax(ctx, input, axis=1)

# LayerNorm
ln = Op.LayerNorm(ctx, input, axis=-1)

# LSTM
lstm_out = Op.LSTM(ctx, input, hidden_size=256, direction="forward")

# 自定义算子
custom_out = Op.Custom(ctx, [input1, input2], ["output1"],
                       "MyCustomOp", '{"param1": 1.0}')
```

#### 模型转换 (Python)

TFDL2提供了Python方式的模型转换工具，相比命令行工具更灵活。

##### ONNX转TFDL2

```python
from TFConvertor import OnnxConvertor
from TFDL2 import DECODER_FLAGS
import glob

# 创建转换器
convertor = OnnxConvertor("my_model")

# 加载ONNX模型
convertor.load("model.onnx", stoptensor=None)

# 运行内置优化（合并BN、融合HardSwish等）
convertor.optmize()

# 构建TFDL2模型图
convertor.buildTFmodel(
    inputshape={"input": [1, 3, 224, 224]},  # 可选，指定输入形状
    std=[0.0039215, 0.0039215, 0.0039215],    # 预处理缩放
    mean=[0, 0, 0]                             # 预处理均值
)

# 可选：验证转换正确性（与ONNX Runtime对比）
if convertor.verification():
    print("转换验证通过！")
    # 保存FP32模型
    convertor.dump("./my_model")
else:
    print("转换验证失败！")

# INT8量化
imglist = glob.glob("./calibration_images/*.jpg")
convertor.quantContext(
    calibration_list=imglist,
    decoderflags=DECODER_FLAGS.RGB,
    MergeConcate=False
)
# 保存量化模型
convertor.dump("./my_model.quant")
```

##### 命令行工具 onnx2tfdl.py

SDK提供了命令行转换脚本 `ConvertTools/python/onnx2tfdl.py`：

```bash
python onnx2tfdl.py \
    --onnxpath model.onnx \
    --output_name my_model \
    --check \
    --quantimgDir ./calibration_images/ \
    --cvtype 1 \
    --mean 0 0 0 \
    --std 0.0039215 0.0039215 0.0039215 \
    --input_shapes input:1,3,224,224
```

| 参数 | 说明 |
|------|------|
| `--onnxpath` | ONNX模型路径（必选） |
| `--output_name` | 输出模型名称（必选） |
| `--stoptensor` | 转换停止节点名称 |
| `--quantimgDir` | 量化校准图片文件夹 |
| `--cvtype` | 图片通道类型：0=BGR, 1=RGB, 2=Gray |
| `--mean` | 预处理均值 |
| `--std` | 预处理标准差 |
| `--input_shapes` | 手动指定输入形状，格式: `name:1,3,H,W` |
| `--check` | 是否验证转换结果（对比ONNX Runtime输出） |

##### 图优化Pass

`Convertor/Optimize/` 目录提供了多种图优化Pass：

| 优化Pass | 说明 |
|----------|------|
| `MergeBatchnorm` | 将BatchNorm折叠到前面的Conv/Scale层 |
| `MergeBiasAdd` | 将独立的Add(bias)合并到Conv中 |
| `MergeHardSwish` | 融合HardSwish激活模式 |
| `MergeHardSigmoid` | 融合HardSigmoid模式 |
| `MergeSwish` | 融合Swish激活模式 |
| `MergeGeLU` | 融合GeLU激活模式 |
| `MergeLayerNorm` | 折叠LayerNorm参数 |
| `MergeBert` | 融合BERT注意力块 |
| `MergeMaskSoftmax` | 融合mask+softmax模式 |
| `Clip2ReLUX` | 将Clip(0, max)转换为ReLUX |
| `PrepareParam` | 预处理ConvTranspose权重、Gemm权重 |
| `replacedilationconv` | 将膨胀卷积替换为等效非膨胀卷积 |
| `SplitConv` | 分割大卷积为多个小操作 |

#### 自定义算子注册 (Python)

```python
from TFDL2.utils import LoadCustomOp, RegisterCustomOp

# 从.so文件加载自定义算子库
LoadCustomOp("libTFDLAddOn.so")

# 注册Python自定义算子（提供reshape和eval函数）
import numpy as np

def my_reshape(input_shapes):
    """根据输入形状推断输出形状"""
    return input_shapes  # 输出与输入同形状

def my_eval(*inputs):
    """执行计算"""
    return np.power(inputs[0], 2)  # 示例：平方操作

RegisterCustomOp("MySquareOp", my_reshape, my_eval)
```

#### 完整推理示例

```python
from TFDL2 import TFContext, TFExecutor, TFTensor
from TFDL2.Common import TFDataType
import numpy as np
import cv2

# 加载模型
ctx = TFContext("yolov5sQ.fb")

# 编译执行体
option = {
    "UseHardware": True,
    "FrugalMode": True,
    "optimize": {"MakeAlign": True, "MakeUnfold": True}
}
executor = TFExecutor(ctx, option)

# 读取并预处理图片
img = cv2.imread("test.jpg")
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_data = img.astype(np.float32) / 255.0
img_data = np.transpose(img_data, (2, 0, 1))
img_data = np.expand_dims(img_data, 0).astype(np.float32)

# 写入输入
inputs = executor.GetInputs()
inputs[0].fromNumpy(img_data)

# 推理
results = executor()

# 处理输出
for output in results:
    raw = output.toNumpy()
    if output.dtype == TFDataType.UINT8:
        # 反量化
        float_data = (raw.astype(np.float32) - output.qzeropoint) * output.qscale
    else:
        float_data = raw
    print(f"{output.name}: shape={float_data.shape}, range=[{float_data.min():.4f}, {float_data.max():.4f}]")
```

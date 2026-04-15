### 卷积

#### **Convloution**

2维卷积操作。该操作将卷积核（Kernel）与输入（Input）间进行卷及操作，结果可选择在加上一个偏置（Bias）后得到输出（Output）。

- **数据形状**：
    - Input：`(batch, channels, rows, cols)`;
    - Kernel：`(filters, channrls, kernel_rows, kernel_cols)`
    - Bias：`(1, filters, 1, 1)`
    - Output：`(batch, filters, new_rows, new_cols)`

- **支持数据类型**：
    - 输入float32，权重float32，偏置float32，输出float32；

    - 输入uint8，权重uint8，偏置int32，输出uint8；

    - 输入uint8，权重uint8，偏置float32，输出uint8；

    - 输入uint8，权重uint8，偏置int32，输出int32；

    - 输入uint8，权重uint8，偏置float32，输出int32；

- **参数列表**：
    - kernel_w：卷积核的宽度；
    - kernel_h：卷积核的高度；
    - stride_w：卷积操作时宽度上的跨度大小；
    - stride_h：卷积操作时高度上的跨度大小；
    - dilation：卷积核的膨胀率，可用于实现膨胀卷积；
    - SAME：padding模式的指示符，若设置为`true`则采用`SAME`的方式进行padding；
    - VALID：padding模式的指示符，若设置为`true`则采用`VALID`的方式进行padding；
    - pad：长度为4的数组，用来表示二维特征上侧、下侧、左侧、右侧的padding宽度。在不使用`SAME`或者`VALID`时可以直接指定各方向上的padding宽度数值；
    - outputChannel：输出通道数，即滤波器数量；
    - hasBias：是否使用偏置；
    - group：卷积分组，用于实现深度可分离卷积（Depthwise Convolution）。

#### **Deconvolution**

2维反卷积操作，相比卷积操作，反卷积会先对输入（Input）进行膨胀操作，然后再与卷积核（Kernel）进行卷积，得到输出（Output），同样可以选择是否增加偏置（Bias）。

- **数据形状：**
    - Input：`(batch, channels, rows, cols)`;
    - Kernel：`(filters, channrls, kernel_rows, kernel_cols)`
    - Bias：`(1, filters, 1, 1)`
    - Output：`(batch, filters, new_rows, new_cols)`

- **支持数据类型：**
    - 输入float32，权重float32，偏置float32，输出float32；

    - 输入uint8，权重uint8，偏置int32，输出uint8；

    - 输入uint8，权重uint8，偏置float32，输出uint8；

- **参数列表：**

    反卷积参数列表继承自卷积参数列表，并且增加一项：

    - outshape：长度为四的数组，指示输出数据形状。

#### **Pooling**

2维池化操作，根据池化类型对输入数据进行池化。

- **数据形状：**
    - Input：`(batch, channels, rows, cols)`;

    - Output：`(batch, channels, new_rows, new_cols)`

- **支持数据类型：**
    - 输入float32，输出float32；

    - 输入uint8，输出uint8；

- **参数列表：**
    - kernel_w：池化核的高度；
    - kernel_h：池化核的宽度；
    - stride_w：池化操作时高度上的跨度大小；
    - stride_h：池化操作时宽度上的跨度大小；
    - dilation：池化时池化核的膨胀比例；
    - ceil_mode：输出特征大小计算时是否使用天花板模式，若设置为`true`则使用；
    - VALID：padding模式的指示符，若设置为`true`则采用`SAME`的方式进行padding；
    - SAME：padding模式的指示符，若设置为`true`则采用`VALID`的方式进行padding；
    - pad：长度为4的数组，用来表示二维特征上侧、下侧、左侧、右侧的padding宽度。在不使用`SAME`或者`VALID`时可以直接指定各方向上的padding宽度数值；
    - poolType：池化模式，可选`MAXPOOL`和`AVGPOOL`，分别对应最大池化和平均池化。
    - global：全局池化模式指示符，若设置为`true`则对整个输入特征图进行池化。

### 核心层

#### **Activation**

激活函数层，将输入数据通过非线性激活函数激活后得到输出。

- **数据形状：**

    输出数据形状均与输入数据形状相同。

- **支持数据类型：**
    - 输入float32，输出float32；

    - 输入uint8，输出uint8；

- **参数列表：**

    - ActivationType：激活函数的类型：

        1. ReLU：
            $$
            y\ =\ \begin{cases}x,\ x\ \geq\ 0\\0,\ x\ <\ 0 \end{cases}
            $$

        2. LeakyReLU:
            $$
            y\ =\ \begin{cases}x,\ x\ \geq\ 0\\ -\alpha\ \times\ x,\ x\ <\ 0 \end{cases}
            $$

        3. PReLU
            $$
            y\ =\ \begin{cases}x,\ x\ \geq\ 0\\ -\alpha_c\ \times\ x,\ x\ <\ 0 \end{cases}
            $$

        4. ReLUX
            $$
            y\ =\ \begin{cases}\alpha,\ x\ >\ \alpha\\x,\ 0\ \leq\ x\ \leq\ \alpha\\0,\ x\ <\ 0 \end{cases}
            $$

        5. Swish
            $$
            y\ =\ x\ ·\ sigmoid(\beta x)
            $$

        6. HardSwish
            $$
            y\ =\ x\ ·\ \frac{ReLU6(x\ +\ 3)} {6}
            $$

        7. Sigmoid
            $$
            y\ =\ \frac {1} {1 + e^{-x}}
            $$

        8. HardSigmoid
            $$
            y\ =\ \frac{ReLU6(x\ +\ 3)} {6}
            $$

        9. Tanh
            $$
            y\ =\ \frac{e^x\ -\ e^{-x}}{e^x\ +\ e^{-x}}
            $$

        10. None：无激活函数

    - threshold：ReLUX的上界。

    - nagativeSlope：LeakyReLU的负半轴斜率。

    - PReLUWeight：PReLU的每通道负半轴斜率。

    - maptable：可使用查表完成的激活函数输入输出对照表。

#### **Flatten**

按照一定维度约束展开输入数据。

- **支持数据类型**

    - 输入float32，输出float32；

    - 输入uint8，输出uint8；

- **参数列表**

    - startAxis：展开开始的维度；
    - endAxis：展开结束的维度。

#### **Flip**

按照宽度和高度翻转输入数据。

- **数据形状**

    输出数据形状均与输入数据形状相同。

- **支持数据类型**

    - 输入float32，输出float32；

    - 输入uint8，输出uint8；

#### **MatMul**

矩阵乘法操作，可以是一个输入（Input）与一个权重（Kernel）之间进行矩阵乘，也可以是两个输入数据之间进行矩阵乘，得到输出（Output），也可以为输出加上一个偏置（Bias）。

- **数据形状**
    - Input：`(m, n)`;
    - Kernel（another Input）：`(n, k)`;
    - Bias：`(1, k)`;
    - Output：`(n, k)`。
- **支持数据类型**
    - 输入float32，权重float32，偏置float32，输出float32；
    - 输入uint8，权重uint8，偏置float32或int32，输出uint8或int32；
    - 输入uint8，权重uint16，偏置float32或int32，输出uint8或int32；
    - 输入uint8，权重bfloat16，偏置bfloat16，输出uint8或int32。
- **参数列表**
    - transA：是否转置输入A的标志位，若设置为`true`则转置输入A；
    - transB：是否转置权重或输入B的标志位，若设置为`true`则转置输入B；
    - InnerProduct：矩阵乘是否满足全连接操作的标志位；
    - hasBias：是否使用偏置；
    - outputChannel：输出数据的通道数。

#### **Mean**

按照参数指示，对对应的维度内的数据取均值并输出。

- **数据形状**
    - Input：`(N, C, H, W)`
    - Output：`(N, C or 1, H or 1, W or 1)`
- **支持数据类型**
    - 输入float32，输出float32；
    - 输入uint8，输出uint8。
- **参数列表**
    - Axis：需要被求平均的维度向量。

#### **Pad**

补零操作，按照参数在输入数据的高度和宽度维度上补零。

- **数据形状**
    - Input：`(N, C, H, W)`；
    - Output：`(N, C, H + pad_t + pad_b, W + pad_l + pad_r)`
- **支持数据类型**
    - 输入float32，输出float32；
    - 输入uint8，输出uint8。
- **参数列表**：
    - pad：长度为4的数组，用来表示二维特征上侧、下侧、左侧、右侧的padding宽度。

#### **Reshape**

按照参数改变输入数据的形状。

- **数据形状**

    - Input：`(N, C, H, W)`；
    - Output：NewShape

- **支持数据类型**：

    输出数据和输入数据类型相同，均可以为TFDL2支持的所有数据类型。

- **参数列表**

    - newshape：输出数据的形状。

#### **Resize**

对输入数据在高度和宽度上进行缩放，缩放可以选择不同的插值方式。

- **数据形状**
    - Input：`(N, C, H, W)`；
    - Output：`(N, C, new_H, new_W)`。
- **支持数据类型**
    - 输入float32，输出float32；
    - 输入uint8，输出uint8。
- **参数列表**
    - interpolationType：插值方式，可以选：
        1. DUPLICATE：重复差值，对于输出为输入size为N倍的操作，在高度和宽度上将输入数据复制N次；
        2. BILINEAR：双线性插值；
        3. NEAREST：最邻近插值。
    - outputHeight：输出数据高度；
    - outputWidth：输出数据宽度；
    - scale：缩放倍数；
    - force_scale：是否按照`scale`来进行缩放，并且忽略outputHeight和outputWidth；
    - align_corners：是否使用角点对齐，若设置为`true`则使用角点对齐。

#### **SetInput**

设置输入参数、对输入数据进行预处理操作。

- **数据形状**

    输出数据与输入数据形状相同

- **支持数据类型**

    输出数据与输入数据类型相同，均可为uint8或float32

- **参数列表**

    - mean：输入参数均值向量，向量长度应与输入数据通道数相同；
    - scale：输入参数缩放尺度向量，向量长度应与输入数据通道数相同；
    - inputshape：输入数据形状；
    - tfDataType：输入数据类型枚举，可以为uint8或float。

#### **Slice**

数据切片操作，可以按照参数在指定维度上将输入数据切割层多份，每份所包含的该维度大小可以由参数设置决定；

- **数据形状**

    - Input：`(N, C, H, W)`；
    - Outputs：`(N, C_i, H, W)`（若在C维度上进行切片操作）。

- **支持数据类型**

    支持所有TFDL2所包含的数据类型，输出数据与输入数据类型相同。

- **参数列表**

    - slicePoint：切片维度上切片的点，向量，长度可以为任意不大于切片维度长度的整数；
    - axis：切片的维度。

#### **Squeeze**

将输入数据的形状中大小为1的维度去除，不改变数据中的内容。

- **数据形状**

    输出形状为输入形状中去除大小为1的维度。

- **支持数据类型**

    支持所有TFDL2所包含的数据类型，输出数据与输入数据类型相同。

#### **Softmax**

松弛最大操作，按照指定维度，将输入中该维度下的值映射到0-1之间，并且保证它们的和为1。

- **数据形状**

    输出数据形状与数据数据形状相同。

- **支持数据类型**

    - 输入float32，输出float32；
    - 输入uint8，输出uint8。

- **参数列表**

    - axis：执行softmax操作的维度。

#### **Transpose**

转置操作，按照参数交换输入数据的各个维度。

- **数据形状**

    - Input：`(M, N)`;

    - Output：`(N, M)`.

    （注：仅为示例，以展示转置对于数据形状的改变）

- **支持数据类型**

    支持所有TFDL2所包含的数据类型，输出数据与输入数据类型相同。

- **参数列表**

    - transAxis：交换后各个维度的顺序向量，长度应当与输入数据的维度数相同。

### 合并层

#### **Concat**

接收一组数据，并将它们按照指定的维度连接起来。

- **数据形状**：

    - Input 1~n：`(N, C, H, W)`

    - Output：`(N, C * n, H, W)`（以channel为拼接维度示例）

- **支持数据类型**

    支持所有TFDL2所包含的数据类型，输出数据与输入数据类型相同。

- **参数列表**

    - axis：被拼接的维度。

#### **Eltwise**

对两个输入数据的内容，按照指定的操作逐一执行，并返回一个数据。操作可以为：相加、相乘、取最大值。

- **数据形状**
    - Input1：`(N, C, H, W)`
    - Input2：`(N, C, H, W)`
    - Output：`(N, C, H, W)`
- **支持数据类型**
    - 输入一float32，输入二float32，输出float32；
    - 输入一uint8，输入二uint8，输出uint8；
    - 输入一int32，输入二int32，输出int32。
- **参数列表**
    - eltwiseOperation：操作指令，可以为：
        1. SUM：逐元素相加操作；
        2. PROD：逐元素相乘操作；
        3. MAX：逐元素取最大操作。
    - broadcastChannel：按通道广播输入指示，若其中一输入的形状为`(N, C, 1, 1)`，则将另一输入每一通道内所有元素与该输入对应通道的数进行操作；
    - broadcaSpatial：按长宽广播输入指示，若其中一输入的形状为`(N, 1, H, W)`，则将另一输入每一个长宽维度内的所有元素与该输入对应长宽位置的数进行操作；
    - eltmode：元素操作硬件执行码，用于判定硬件加速的方式。

### 规范化层

#### **BatchNormal**

批规范化操作。 该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1 。

- **数据形状**
    - Input：`(N, C, H, W)`;
    - mean：`(1, C, 1, 1)`;
    - var：`(1, C, 1, 1)`;
    - Output：`(N, C, H, W)`
- **支持数据类型**
    - 输入float32，mean float32，var float32，输出float32；
    - 输入uint8，mean float32， var float32， 输出uint8；

#### **L2Normalization**

L2归一化操作，对于一个长度为D的向量，其归一化公式如下：
$$
y_i\ =\ \frac {x_i} {\sqrt{\sum_{i=1}^Dx_i^2}}
$$

- **数据形状**
    - Input：`(N, C, H, W)`；
    - Output：`(N, C, H, W)`
- **支持数据类型**
    - 输入float32，输出float32；
    - 输入uint8，输出uint8。

#### **LocalResponseNormalization**

局部响应归一化层，在指定范围内对activation进行归一化，出现在训练中以防止过拟合。
$$
b^i_{x,\ y}\ =\ a^i_{x,\ y}\ /\ (k\ +\ \alpha\sum^{min(N-1,\ i+n/2)}_{j=max(0,\ i-n/2)}(a^j_{x,\ y})^2)^\beta
$$

- **数据形状**
    - Input：`(N, C, H, W)`；
    - Output：`(N, C, H, W)`
- **支持数据类型**
    - 输入float32，输出float32；
    - 输入uint8，输出uint8。
- **参数列表**
    - size：局部响应归一化的计算半径大小；
    - alpha：上述公式中的α值；
    - beta：上述公式中的β值。

#### **Scale**

尺度变换层，对输入每个通道内的值进行线性尺度缩放。
$$
y_{c,\ i,\ j}\ =\ x_{c,\ i,\ j}\ ·\ \gamma_c\ +\ \beta_c
$$

- **数据形状**
    - Input：`(N, C, H, W)`；
    - Output：`(N, C, H, W)`
- **支持数据类型**
    - 输入float32，beta float32，gamma float32，输出float32；
    - 输入uint8，beta float32，gamma float32，输出uint8。

### 量化层

#### **Dequantize**

反量化层，将输入的uint8数据反量化成float数据并输出。

#### **Quantize**

量化层，将输入的float数据量化成uint8数据并输出。

### 高级层

#### **CropAndResize**

从输入特征图中按照指定的边界框（bounding box）裁剪区域并缩放到固定大小输出。常用于R-CNN系列目标检测的RoI处理。

- **数据形状**：
    - Image：`(batch, channels, height, width)`
    - Boxes：`(num_boxes, 4)`，格式为 `[y1, x1, y2, x2]`，值域 `[0, 1]`（归一化坐标）
    - BoxIndex：`(num_boxes,)` 指定每个box属于哪个batch
    - Output：`(num_boxes, channels, crop_height, crop_width)`

- **支持数据类型**：
    - 输入float32，输出float32；
    - 输入uint8，输出uint8。

- **参数列表**：
    - cropHeight：裁剪后输出高度；
    - cropWidth：裁剪后输出宽度；
    - extrapolationValue：外推填充值，默认为0。

#### **DetectionOutput**

SSD目标检测的后处理层。将预测的边界框、置信度和先验框（PriorBox）结合，执行NMS（非极大值抑制），输出最终的检测结果。

- **数据形状**：
    - Localization：`(batch, num_priors * 4)` — 边界框回归预测
    - Confidence：`(batch, num_priors * num_classes)` — 类别置信度
    - PriorBox：`(2, num_priors * 4)` — 先验框及其方差
    - Output：`(1, 1, num_detections, 7)` — 每行格式为 `[batch_id, class_id, score, x1, y1, x2, y2]`

- **支持数据类型**：
    - 输入float32，输出float32。

- **参数列表**：
    - numClasses：检测类别数量；
    - shareLocation：所有类别是否共享边界框预测；
    - backgroundLabelId：背景类别ID（通常为0）；
    - nmsThreshold：NMS的IoU阈值；
    - confidenceThreshold：置信度过滤阈值；
    - topK：每类最多保留的检测数；
    - keepTopK：最终保留的检测总数；
    - varianceEncodedInTarget：方差是否编码在目标中。

#### **Normalize**

L2归一化层，对输入数据沿指定通道维度进行L2归一化。常用于SSD中的特征归一化。

- **数据形状**：
    - Input：`(batch, channels, height, width)`
    - Output：`(batch, channels, height, width)`（形状不变）

- **支持数据类型**：
    - 输入float32，输出float32；
    - 输入uint8，输出uint8。

- **参数列表**：
    - acrossSpatial：是否在整个空间维度上归一化（跨H和W）；
    - channelShared：是否所有通道共享同一个缩放参数。

#### **Power**

对输入数据进行幂运算：`y = (shift + scale * x) ^ power`。常用于Caffe模型中的数据预处理。

- **数据形状**：
    - 输出与输入形状相同。

- **支持数据类型**：
    - 输入float32，输出float32；
    - 输入uint8，输出uint8（内部使用查找表加速）。

- **参数列表**：
    - power：指数；
    - scale：缩放系数；
    - shift：偏移量。

- **INT8加速**：当输入输出均为uint8时，Power算子在编译期会构建一个256项查找表（lookup table），推理时使用查表代替浮点运算，大幅提升速度。

#### **PriorBox**

SSD中的先验框（Anchor）生成层。根据特征图的尺寸、步长、最小/最大尺度等参数生成一系列先验框。

- **数据形状**：
    - Feature Map：`(batch, channels, height, width)` — 当前层特征图
    - Image Data：`(batch, channels, img_height, img_width)` — 原始图像
    - Output：`(2, height * width * num_priors * 4)` — 先验框坐标 + 方差

- **支持数据类型**：
    - 输入float32，输出float32。

- **参数列表**：
    - minSize：最小尺度（像素）；
    - maxSize：最大尺度（像素）；
    - aspectRatio：宽高比列表（如 `[1.0, 2.0, 0.5]`）；
    - variance：先验框方差，长度为4的向量；
    - flip：是否对宽高比进行翻转；
    - clip：是否将先验框裁剪到图像范围内；
    - step：特征图步长（宽和高）；
    - offset：先验框中心偏移量，通常为0.5。

### 包装器

#### **OnnxWrapper**

ONNX算子包装器，用于包装TFDL2原生不支持的ONNX算子。在模型转换时，对于无法直接映射到TFDL2算子的ONNX节点，会将其整体包装为一个OnnxWrapper节点，在运行时调用ONNX Runtime执行该节点。

- **数据形状**：与被包装的原始ONNX节点一致
- **支持数据类型**：float32
- **说明**：此包装器保证了模型转换的兼容性，但性能不如原生TFDL2算子。建议对性能敏感的算子编写TFDL2自定义算子替代。

#### **TensorflowWrapper**

TensorFlow算子包装器，功能类似OnnxWrapper，用于包装TFDL2原生不支持的TensorFlow算子。

- **数据形状**：与被包装的原始TensorFlow节点一致
- **支持数据类型**：float32
- **说明**：同OnnxWrapper，仅保证功能兼容性。

### 专用层

#### **Align**

内存对齐层，用于在NPU硬件执行前对张量的内存布局进行对齐处理。当NPU要求数据在特定边界上对齐（如128字节对齐）时，编译器会自动插入Align节点。

- **数据形状**：
    - Input：`(batch, channels, height, width)`
    - Output：`(batch, channels, height, aligned_width)`（空间维度可能因对齐而扩大）

- **支持数据类型**：
    - 支持所有TFDL2数据类型，输出与输入类型相同。

- **参数列表**：
    - alignSize：对齐粒度（字节数）。

- **说明**：Align层通常由编译器在`MakeAlign`优化选项开启时自动插入，用户无需手动添加。

#### **ZeroMask**

零值掩码层，将输入中值为零的元素替换为指定的非零值（或将非零元素置零）。用于注意力机制中的mask处理。

- **数据形状**：
    - 输出与输入形状相同。

- **支持数据类型**：
    - 输入float32，输出float32；
    - 输入uint8，输出uint8。

- **参数列表**：
    - maskValue：掩码值，用于替换零值位置的数值。

### 扩展神经网络算子

以下算子在新版SDK中支持，通过TFModule.h的`TFFunc`命名空间或Python `Op`模块使用。

#### **LSTM**

长短时记忆网络层。

- **数据形状**：
    - Input：`(seq_len, batch, input_size)`
    - Output：`(seq_len * num_directions, batch, hidden_size)`

- **参数列表**：
    - hiddenSize：隐藏层大小；
    - direction：方向，`"forward"`、`"backward"`或`"bidirectional"`。

#### **Einsum**

爱因斯坦求和约定操作，支持通用的张量运算。

- **参数列表**：
    - equation：Einsum表达式字符串，如`"bhqk,bkhd->bqhd"`。

#### **WarpWithOpFlow / WarpWithAffine / WarpWithPerspect**

仿射变换和透视变换操作，用于空间变换网络（STN）和光流对齐。

- **WarpWithOpFlow**：使用光流场进行图像变形
- **WarpWithAffine**：使用仿射变换矩阵进行变形
- **WarpWithPerspect**：使用透视变换矩阵进行变形

#### **GreedyCTC**

CTC（连接时序分类）贪心解码器。用于语音识别和OCR中的序列解码。

- **数据形状**：
    - Input：`(seq_len, batch, num_classes)`
    - Output：`(batch, max_seq_len)` — 解码后的类别序列

#### **Unfold**

将滑动窗口展开操作（im2col），用于某些特殊卷积实现。

- **参数列表**：
    - kernel_w / kernel_h：窗口大小；
    - stride_w / stride_h：步长；
    - pad：填充。

#### **LayerNorm**

层归一化，对输入张量沿指定轴进行归一化。

- **数据形状**：输出与输入形状相同
- **参数列表**：
    - axis：归一化的轴；
    - gamma / beta：可学习的缩放和偏移参数。

#### **GeLU**

高斯误差线性单元激活函数。

$$
y = x \cdot \Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。

#### **Mish**

Mish激活函数。

$$
y = x \cdot \tanh(\ln(1 + e^x))
$$

#### **ELU**

指数线性单元。

$$
y = \begin{cases}x, & x \geq 0 \\ \alpha(e^x - 1), & x < 0 \end{cases}
$$

- **参数列表**：
    - alpha：负半轴的缩放系数，默认为1.0。
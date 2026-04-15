### TFCV 图像与视频读写模块

TFCV是TFDL2 SDK中提供的计算机视觉IO模块，封装了图片解码、视频流读取、RTSP流接入等功能。TFCV内部会根据SDK运行环境自动选择硬件解码（基于TFG）或软件解码（基于OpenCV）。

#### 命名空间

所有TFCV函数位于 `TFDL_CAPI::TFCV` 命名空间下。

#### 数据类型枚举

```cpp
enum DECODER_FLAGS {
    TFCV_BGR  = 0,   // BGR通道顺序
    TFCV_RGB  = 1,   // RGB通道顺序
    TFCV_Gray = 2,   // 灰度图
    TFCV_I420 = 3    // I420原始格式
};

enum MediaTYPE {
    tJPEG = 0,   // JPEG图片
    tPNG  = 1,   // PNG图片
    tBMP  = 2,   // BMP图片
    tH264 = 3,   // H264视频流
    tHEVC = 4    // H265/HEVC视频流
};
```

#### 核心API

##### 创建Reader

```cpp
// 创建图片读取器
TFVision NewImgReader();

// 创建视频读取器
// type: 媒体类型 (MediaTYPE)
// decoderpath: 硬件解码器设备路径，如 "/dev/mv500"
//   NPU40T设备上可用的解码器路径：
//   "/dev/mv500"   - 第一个解码器
//   "/dev/mv500-2" - 第二个解码器
//   "/dev/mv500-3" - 第三个解码器
// 若使用软件解码，decoderpath传空字符串
TFVision NewVideoReader(int type, string decoderpath);
```

##### 打开媒体源

```cpp
// 从文件路径或RTSP地址打开
void OpenURL(TFVision vision, string url);

// 从内存中打开（适用于从网络接收的JPEG图片流）
void OpenSource(TFVision vision, uint8_t* data, size_t len);

// 关闭媒体源
void Close(TFVision vision);

// 拷贝Vision对象（共享内部状态）
void VisionCopy(TFVision dst, TFVision src);
```

##### 读取与解码

```cpp
// 读取视频下一帧，返回值：
//   1 = 成功读取
//   0 = 流已结束
//  -1 = 读取错误
int ReadFrame(TFVision vision);

// 将当前帧解码到张量中（图片Reader专用）
// 图片Reader在OpenSource/OpenURL后自动完成解码
bool DumpImgData(TFVision vision, TFTensor dst,
                 int batch = 0,
                 DECODER_FLAGS flags = TFCV_BGR);

// 将当前帧解码到vector<uint8_t>缓冲区
bool DumpImgData(TFVision vision, vector<uint8_t>& dst,
                 DECODER_FLAGS flags = TFCV_BGR,
                 float scale = 1.0,
                 CropSize cropSize = {0, 0, 0, 0});

// 将缓冲区数据压缩为JPEG
bool Compress(TFVision vision, string ext,
              vector<uint8_t>& dst, float scale = 1.0);
```

##### 视频属性

```cpp
int GetWidth(TFVision vision);     // 获取帧宽度
int GetHeight(TFVision vision);    // 获取帧高度
float GetFps(TFVision vision);     // 获取视频帧率
void SetFps(TFVision vision, float fps);  // 设置目标帧率
MediaTYPE GetType(TFVision vision); // 获取媒体类型
```

##### 初始化

```cpp
// 初始化TFCV子系统（在使用视频功能前调用）
void TFCV_INIT();
```

#### 使用示例

##### 图片分类完整流程

```cpp
#include "TFDL2_C_API.h"
#include "TFCV/TFCV.h"
using namespace TFDL_CAPI;

// 1. 加载模型并编译执行体
auto context = LoadProto("ResNet50.fb");
auto executor = CompileExecutor(context, true, "{\"UseHardware\":true,\"FrugalMode\":true}");

// 2. 获取输入张量
auto inputs = GetInputTensors(executor);

// 3. 创建图片读取器并加载图片
TFVision reader = TFCV::NewImgReader();

// 方式A：从文件加载
TFCV::OpenURL(reader, "test.jpg");

// 方式B：从内存加载（适用于网络传输的图片流）
std::fstream file("test.jpg", std::ios::binary);
std::string imgstr((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
std::vector<uint8_t> imgdata(imgstr.begin(), imgstr.end());
TFCV::OpenSource(reader, imgdata.data(), imgdata.size());

// 4. 解码图片到张量（自动缩放到模型输入尺寸）
TFCV::DumpImgData(reader, inputs[0], 0, TFCV::TFCV_RGB);

// 5. 推理
ForwardExecutorAlone(executor);

// 6. 读取输出
auto outputs = GetOutputTensors(executor);
float* result = (float*)GetTensordata(outputs[0]);
```

##### RTSP视频流处理

```cpp
#include "TFDL2_C_API.h"
#include "TFCV/TFCV.h"
using namespace TFDL_CAPI;

// 1. 加载模型
auto context = LoadProto("yolov5sQ.fb");
auto executor = CompileExecutor(context, true, "{\"UseHardware\":true,\"FrugalMode\":true}");
auto inputs = GetInputTensors(executor);
auto outputs = GetOutputTensors(executor);

// 2. 创建视频读取器，使用NPU硬件解码器
TFVision video = TFCV::NewVideoReader(TFCV::tH264, "/dev/mv500");
TFCV::OpenURL(video, "rtsp://admin:password@192.168.1.100:554/stream");

// 3. 设置帧率（FPS=1时内部只解码I帧，适合多路场景）
TFCV::SetFps(video, 25);

// 4. 循环读取帧并推理
while (true) {
    int ret = TFCV::ReadFrame(video);
    if (ret == 0) break;     // 流结束
    if (ret == -1) continue; // 读取错误，跳过

    // 解码当前帧到输入张量
    TFCV::DumpImgData(video, inputs[0], 0, TFCV::TFCV_RGB);

    // 推理
    ForwardExecutorAlone(executor);

    // 处理输出...
    uint8_t* data = (uint8_t*)GetTensordata(outputs[0]);
    // ... 后处理逻辑
}

TFCV::Close(video);
```

##### 多路视频并发

```cpp
// NPU40T有多个硬件解码器，可以同时处理多路视频
// 每个视频流对应一个独立的解码器和执行体
auto context = LoadProto("yolov5sQ.fb");

std::vector<TFExecutor> executors;
std::vector<TFVision> videos;

for (int i = 0; i < 4; i++) {
    // 每个执行体绑定不同的NPU核心
    std::string config = "{\"UseHardware\":true,\"FrugalMode\":true,\"Core\":[-1]}";
    auto exec = CompileExecutor(context, true, config);
    executors.push_back(exec);

    // 使用不同的硬件解码器
    std::string dev = "/dev/mv500" + (i > 0 ? ("-" + std::to_string(i+1)) : "");
    TFVision vid = TFCV::NewVideoReader(TFCV::tH264, dev);
    TFCV::OpenURL(vid, rtsp_urls[i]);
    TFCV::SetFps(vid, 15);
    videos.push_back(vid);
}
```

#### 注意事项

1. **图片解码**：图片读取器在调用`OpenURL`或`OpenSource`后即完成解码，之后调用`DumpImgData`将数据写入张量。若需要处理多张图片，需要重新调用`OpenURL`/`OpenSource`。
2. **视频帧率控制**：`SetFps`可以控制视频读取的帧率。当FPS设置为1时，TFCV内部会只解码I帧（关键帧），这对于安防摄像头的多路并发场景非常有用，可以大幅降低解码开销，支持更多路数。
3. **硬件解码器数量**：NPU40T设备上通常有多个硬件解码器（如`/dev/mv500`、`/dev/mv500-2`、`/dev/mv500-3`），每个解码器可以独立处理多路视频流。
4. **颜色格式**：`DumpImgData`的`flags`参数决定输出的通道顺序。模型训练时的输入格式应与此保持一致（大多数OpenCV训练的模型使用BGR，大部分PyTorch/ONNX模型使用RGB）。
5. **内存生命周期**：`TFVision`使用引用计数管理内部资源，可以安全地拷贝和传递。当最后一个引用销毁时自动释放资源。
6. **TFCV初始化**：在使用视频功能前需要调用`TFCV_INIT()`进行子系统初始化。图片功能无需此调用。

### 多线程与多执行体编程指南

TFDL2的架构设计将网络描述（TFContext）与运行时执行体（TFExecutor）分离，使得多线程并发推理成为可能。本文档介绍如何在生产环境中高效使用多执行体模式。

---

## 核心概念

### TFContext 与 TFExecutor 的关系

```
TFContext (网络结构 + 权值)
    │
    ├── CompileExecutor(context, true, config1) → TFExecutor #1 (核心0)
    ├── CompileExecutor(context, true, config2) → TFExecutor #2 (核心1)
    ├── CompileExecutor(context, true, config3) → TFExecutor #3 (核心2)
    └── CompileExecutor(context, true, config4) → TFExecutor #4 (核心3)
```

- **TFContext**：只读的网络描述，包含拓扑结构和权值数据，不分配运行时内存
- **TFExecutor**：编译后的执行体，包含运行时内存和命令队列，执行实际推理
- **权值共享**：`CompileExecutor(context, true, config)` 的第二个参数 `shareWeight=true` 时，所有执行体共享同一个Context的权值内存，大幅节省内存

### 两种推理模式

#### ForwardExecutorAlone — 独立推理（推荐）

```cpp
// 独立执行推理，线程安全，不参与调度器协调
ForwardExecutorAlone(executor);
```

- 线程安全：可以多个线程同时调用不同执行体的`ForwardExecutorAlone`
- 适合：多路并发推理，每路独立运行
- **推荐在大多数场景使用**

#### ForwardExecutor — 协调推理

```cpp
// 由全局调度器协调执行顺序
ForwardExecutor(executor);
```

- 调度器会根据多个执行体的运行状态动态调整执行顺序，缓解CPU与NPU之间的等待
- 通过 `SetForwardWindowSize(n)` 设置调度窗口大小
- 适合：需要精细调度的高吞吐场景

---

## 多执行体并发模式

### 模式1：多路独立推理

每路视频流对应一个独立的执行体，绑定到不同的NPU核心。

```cpp
#include "TFDL2_C_API.h"
#include "TFCV/TFCV.h"
#include <thread>
#include <vector>
using namespace TFDL_CAPI;

void inference_worker(TFExecutor executor, TFVision video) {
    auto inputs = GetInputTensors(executor);
    auto outputs = GetOutputTensors(executor);

    while (true) {
        int ret = TFCV::ReadFrame(video);
        if (ret != 1) break;

        TFCV::DumpImgData(video, inputs[0], 0, TFCV::TFCV_RGB);
        ForwardExecutorAlone(executor);

        // 处理输出...
    }
}

int main() {
    auto context = LoadProto("yolov5sQ.fb");
    std::vector<TFExecutor> executors;
    std::vector<TFVision> videos;
    std::vector<std::thread> threads;

    // 创建4路执行体
    for (int i = 0; i < 4; i++) {
        // 每个执行体绑定不同核心
        std::string config = "{\"UseHardware\":true,\"FrugalMode\":true,\"Core\":[-1]}";
        auto executor = CompileExecutor(context, true, config);
        executors.push_back(executor);

        TFVision video = TFCV::NewVideoReader(TFCV::tH264, "/dev/mv500");
        TFCV::OpenURL(video, rtsp_urls[i]);
        TFCV::SetFps(video, 25);
        videos.push_back(video);
    }

    // 启动工作线程
    for (int i = 0; i < 4; i++) {
        threads.emplace_back(inference_worker, executors[i], videos[i]);
    }

    for (auto& t : threads) t.join();
    return 0;
}
```

### 模式2：流水线推理（检测+跟踪）

多个执行体串联，形成推理流水线。使用 `moodycamel::ConcurrentQueue` 等线程安全队列传递中间结果。

```cpp
// 线程1：检测
auto det_context = LoadProto("yolov5sQ.fb");
auto det_executor = CompileExecutor(det_context, true, det_config);

// 线程2：ReID特征提取
auto reid_context = LoadProto("vehicleReidQ.fb");
auto reid_executor = CompileExecutor(reid_context, true, reid_config);

// 检测线程
void detection_worker(/* ... */) {
    auto inputs = GetInputTensors(det_executor);
    auto outputs = GetOutputTensors(det_executor);

    while (running) {
        // 获取输入图片
        ForwardExecutorAlone(det_executor);

        // 后处理（NMS等）获取检测框
        auto detections = postprocess(outputs);

        // 将裁剪的目标区域送入ReID队列
        for (auto& det : detections) {
            reid_queue.push(crop_image(input, det.bbox));
        }
    }
}

// ReID线程
void reid_worker(/* ... */) {
    auto inputs = GetInputTensors(reid_executor);
    auto outputs = GetOutputTensors(reid_executor);

    while (running) {
        cv::Mat crop;
        reid_queue.pop(crop);

        // 写入裁剪区域
        write_to_tensor(inputs[0], crop);
        ForwardExecutorAlone(reid_executor);

        // 获取特征向量
        float* features = (float*)GetTensordata(outputs[0]);
    }
}
```

### 模式3：批量推理

对于小模型（如ReID、分类），可以将多个输入合并为一个batch以提高吞吐量。

```cpp
auto context = LoadProto("vehicleReidQ.fb");

// 编译时设置较大的batch
std::string config = "{\"UseHardware\":true,\"FrugalMode\":true,"
    "\"InputShape\":[{\"NodeName\":\"input.1\",\"Shape\":[8,3,64,64]}]}";
auto executor = CompileExecutor(context, true, config);

auto inputs = GetInputTensors(executor);
auto outputs = GetOutputTensors(executor);

// 将8个目标区域写入同一个batch
for (int i = 0; i < 8; i++) {
    write_crop_to_batch(inputs[0], crops[i], i);
}

ForwardExecutorAlone(executor);

// 从输出中提取8个特征向量
float* data = (float*)GetTensordata(outputs[0]);
for (int i = 0; i < 8; i++) {
    std::vector<float> feature(data + i * feat_dim, data + (i + 1) * feat_dim);
}
```

---

## NPU核心分配策略

### NPU10T

- 每个设备只有1个NPU核心
- 每个执行体独占一个核心
- 多路并发需要多个NPU10T设备

### NPU40T

- 每个设备有多个NPU核心（通常4个或更多）
- 支持单执行体多核心（自动并行）
- 支持多执行体各绑定不同核心

```cpp
// 大核-小核模式
config["Core"] = json11::Json::array{-1};  // 大核，高性能
config["Core"] = json11::Json::array{-2};  // 小核，低功耗

// 指定核心编号
config["Core"] = json11::Json::array{0};
config["Core"] = json11::Json::array{1};

// 多核心（单执行体使用多核心）
config["Core"] = json11::Json::array{0, 1, 2, 3};
```

**推荐策略：**
- 高QPS场景：多执行体 + 多核心，每个执行体绑定一个核心
- 低延迟场景：单执行体多核心
- 混合场景：检测模型绑定大核，轻量模型绑定小核

---

## 内存管理

### 权值共享

```cpp
auto context = LoadProto("model.fb");

// shareWeight=true: 共享权值，节省内存
auto exec1 = CompileExecutor(context, true, config1);
auto exec2 = CompileExecutor(context, true, config2);

// shareWeight=false: 复制权值，每个执行体独立
auto exec3 = CompileExecutor(context, false, config3);
```

**重要**：`shareWeight=true`时，不能释放TFContext，直到所有使用它的执行体都销毁。

### FrugalMode

开启`FrugalMode`后，执行体会在推理过程中复用中间张量内存。对于深层网络，可以将内存占用降低到原来的30%-50%。

### 内存生命周期

```cpp
{
    auto context = LoadProto("model.fb");  // 加载权值
    {
        auto executor = CompileExecutor(context, true, config);
        // executor拥有运行时内存（输入/输出缓冲区、命令队列等）
        // executor与context共享权值内存

        ForwardExecutorAlone(executor);
        auto outputs = GetOutputTensors(executor);
        // outputs的内存属于executor，在executor销毁前有效
    }
    // executor销毁，运行时内存释放
    // context仍存在，权值内存仍可用
}
// context销毁，权值内存释放
```

---

## 线程安全规则

1. **TFContext是只读的**：多个线程可以同时读取同一个TFContext
2. **TFExecutor是线程局部的**：同一个TFExecutor不能被多个线程同时使用
3. **TFTensor属于其Executor**：通过`GetInputTensors`/`GetOutputTensors`获取的张量，其内存在对应Executor的生命周期内有效
4. **ForwardExecutorAlone是线程安全的**：不同Executor的`ForwardExecutorAlone`可以并发调用
5. **输入写入必须在Forward之前**：写入输入张量和调用推理不应并发进行

---

## 性能调优建议

1. **执行体数量**：通常等于NPU核心数量。过多的执行体会导致CPU开销增加和内存争用。
2. **预处理分离**：将图片解码和预处理放在独立线程中，避免阻塞推理线程。
3. **实时调度**：对于实时性要求高的场景，使用`SCHED_FIFO`调度策略：
   ```cpp
   struct sched_param param;
   param.sched_priority = sched_get_priority_max(SCHED_FIFO);
   pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
   ```
4. **视频I帧模式**：多路场景下设置`FPS=1`让TFCV只解码I帧，减少解码开销。
5. **批量合并**：对于小模型，将多路输入合并为一个batch推理，减少推理调用次数。

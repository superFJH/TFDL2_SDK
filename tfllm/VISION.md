# SAM 与 Depth Anything

`tfllm.vision` 提供 Python 加载和推理入口，复用 TFLLM 的原生 Linear、空间卷积和
融合 Attention。官方模型负责提示编码、位置编码、视频状态、DPT head、插值及任务后处理。
这是 PyTorch 模型适配器，不是原生 C++ 模型包，也不读取 TFDL `.fb`。

| family | 权重 / provider | 入口 |
|---|---|---|
| `sam` | 官方 `.pth` + `vit_b/vit_l/vit_h`；本地 HF 目录 | 图像点/框提示，缓存 image embeddings |
| `sam2` | 官方 `.pt` + model config（含 2.1）；本地 HF 目录 | 图像提示；官方 provider 的视频状态/传播 |
| `sam3` | 官方 checkpoint + tokenizer 资源；本地 HF 目录 | 图像文字提示；官方 processor 的几何提示 |
| `efficienttam` | 官方 `.pt` + model config | 图像提示、视频状态/传播 |
| `depth-anything` | 本地 HF V1 目录；或 `attach` 已加载的官方模型 | 相对深度 |
| `depth-anything-v2` | 官方 `.pth` + encoder；本地 HF V2/metric 目录 | 深度图，保留 checkpoint 的 relative/metric 语义 |
| `depth-anything-v3` | 官方本地模型目录 | 单图/多视图 `inference`，保留深度、置信度和相机结果 |

SAM3 官方 multi-GPU 视频 predictor 依赖 CUDA，当前加载器明确拒绝该入口。
V3 的多视图推理不是 SAM 的时序目标跟踪，也不保证所有 Gaussian Splatting/export 分支适合 CPU。
其它 SAM 衍生架构不自动视为上述模型；不能仅凭名称判定兼容。

## 安装

编译并安装 `TFLLM_WITH_TORCH=ON`。开发目录设置：

```sh
export PYTHONPATH="$PWD/TFLLM/python"
export TFLLM_TORCH_LIBRARY="$PWD/build/tfllm/libtfllm-torch.so"
```

macOS 使用对应 `.dylib`。在安装后的 `SDK/tfllm` 中可执行
`python -m pip install '.[vision]'`。此 extra 安装 Torch、TorchVision、Transformers、
NumPy、Pillow、OpenCV；官方 provider 的 SAM/SAM2/SAM3/EfficientTAM/DA2/DA3 包需按各自项目安装，
不会自动下载 checkpoint。不同官方项目的 Python、Torch 和 optional dependency 要求仍适用。
本地 HF 适配针对 Transformers 5.19；SAM3 官方包的要求不能用 HF 测试结果代替。

## 加载与调用

默认使用 NPU，硬件初始化/执行失败直接报错。`backend="cpu"` 是显式的开发参考路径。
默认 `local_files_only=True`；只有显式设置为 false 才允许 HF 下载。

```python
import numpy as np
from PIL import Image
from tfllm.vision import load

image = np.asarray(Image.open("frame.jpg").convert("RGB"))
with load("sam2", "/models/sam2.1_hiera_tiny.pt",
          config="configs/sam2.1/sam2.1_hiera_t.yaml") as sam:
    result = sam.segment(image, point_coords=np.array([[240, 180]]),
                         point_labels=np.array([1]))
    masks, scores = result["masks"], result["scores"]
    print(sam.report.to_dict())

# For repeated prompts, preserve the same image features:
# sam.set_image(image)
# masks, scores, low_res = sam.predict(point_coords=..., point_labels=...)
```

`segment` 的点与框使用原图坐标；官方 SAM/SAM2/EfficientTAM 输入为 RGB NumPy 图。
HF 的 `predict` 保留其 `input_points/input_labels/input_boxes` 参数；方便入口 `segment` 会构造批次。
官方 predictor 方法可直接通过 session 调用，不重建 prompt/video state。

```python
with load("sam3", "/models/sam3.pt", model_options={"bpe_path": "/models/bpe_simple_vocab_16e6.txt.gz"}) as sam:
    result = sam.segment(Image.open("frame.jpg").convert("RGB"), text_prompt="person")
    # Geometric prompts use the official processor API and its original state:
    # state = sam.set_image(image)
    # sam.add_geometric_prompt(..., state=state)

with load("efficienttam", "/models/efficienttam_s_512x512.pt",
          config="configs/efficienttam/efficienttam_s_512x512.yaml", task="video") as tracker:
    state = tracker.init_state(video_path="/videos/frames")
    tracker.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                 points=np.array([[240, 180]]), labels=np.array([1]))
    for frame_idx, object_ids, mask_logits in tracker.propagate_in_video(state):
        pass  # Consume each frame's result before closing the session.
```

Video generator 的每次 `next` 单独进入 inference/routing scope；暂停迭代时不会改变调用者的
Torch grad 或 SDPA 行为。每个 session 串行保护 predictor state；跨 session 的 runtime 可共享。

官方 SAM2Base/Sam3Image 的 attention 路由限制在模型模块内，避免为整张图的所有 Torch
操作安装 TorchFunctionMode。Hiera 的 Q pooling、SAM3 的 RoPE/relative position 和原始
predictor 调用保持原语义；未识别模型保留 session scope。
在 A77 上可使用 `NpuRuntime(attention_policy="hybrid", attention_fuse_workers=32,
attention_score_mib=16)` 交给 `load(..., runtime=runtime)`，按外部 dtype/shape 选择
NPU 大 attention 与 Torch CPU 小窗口。详细边界见 [TORCH.md](TORCH.md)。

FAST 是另一个显式可选的通用 attention 算法，可直接设置
`TFLLM_NPU_ATTENTION_FAST=1` 后运行现有 `vision.load/attach`，或传入
`NpuRuntime(attention_fast=True)`。FAST 会覆盖上述 hybrid 的 CPU 小窗口分流，
并在整个 session 调用中接管 SDPA；SAM3 packed MHA 的逐头 mask/bias 也保留。
FP16/FP32、self/cross、window 与单 query 共用同一内核，head dimension ≤256、
keys ≤16384；任意 bool/浮点 bias 与全屏蔽行受支持。不支持的 attention 语义
明确报错。Linear/卷积/FFN 的分工与 predicate 保持原配置。

```python
from tfllm.torch import NpuRuntime
from tfllm.vision import load
with NpuRuntime(attention_fast=True, attention_policy="hybrid") as rt:
    with load("sam2", "/models/sam2.1_hiera_tiny.pt", runtime=rt,
              config="configs/sam2.1/sam2.1_hiera_t.yaml") as sam:
        pass  # sam.segment / set_image / predict 保持原接口
```

FAST 默认关闭，短窗口可能因额外变换/调度而变慢。现有最快实测配置与本开关
需分别评估；`vision_fast_inference.py --engine auto` 在环境 FAST=1 时会选 TFLLM，
包括 EfficientTAM；显式 `--engine torch` 仍表示调用者选择原 Torch。
数值及覆盖记录见[通用 FAST 报告](artifacts/attention-general-fast-20261017-server65/REPORT.md)。

`attach()` / `load()` 同时启用快速逐点激活的自动策略，按 dtype/元素数选择原生核或 Torch。
可设置 `activation_policy="torch"` 对照关闭激活加速，或 `"native"` 显式测试原生核；
`predicate(path, module)` 可选择需要接管的投影、卷积和激活，默认不额外排除模块。
FP16 需由调用者显式使用 CPU autocast；加载器保持 checkpoint 的原始精度。
整图编码和交互提示应分别测量，激活微基准收益不能直接当作模型加速比。

```python
import cv2

with load("depth-anything-v2", "/models/depth_anything_v2_vits.pth", encoder="vits") as depth:
    prediction = depth.infer_image(cv2.imread("frame.jpg"), input_size=518)
    # HxW floating depth, not a min-max-normalized visualization.

with load("depth-anything", "/models/depth-anything-small-hf") as depth:
    prediction = depth.infer_image(cv2.imread("frame.jpg"))

with load("depth-anything-v3", "/models/DA3-LARGE") as depth:
    prediction = depth.inference(["frame1.jpg", "frame2.jpg"])
    # prediction.depth / conf / extrinsics / intrinsics retain the official API.
```

V1/V2 的 `infer_image` 接受 BGR；官方 provider 保留 OpenCV bicubic resize、14 倍数对齐、
ImageNet normalization 和 `align_corners=True` 输出插值。HF provider 保留其 processor 的
预处理/后处理，因此与官方 OpenCV 入口不承诺逐像素相同。V3 保留官方多视图 API 和相机对齐流程。

已加载的官方模型可直接接入，无需再次读权重：

```python
from tfllm.torch import NpuRuntime
from tfllm.vision import attach

with NpuRuntime() as runtime:
    with attach(model.cpu().eval(), family="sam2", runtime=runtime, processor=predictor) as session:
        session.set_image(image)
        result = session.predict(point_coords=points, point_labels=labels)
```

session 不关闭外部传入的 runtime；`load` 创建的 runtime 在 session.close 时释放。
不要并发改权重、使用 `.data` 绕过版本计数、在关闭后继续迭代，或直接调用原 processor 来绕过 session。
关闭后恢复原 forward；原 Parameter、state_dict 和共享权重不被替换。
加载权重需在 inference_mode 外进行；`attach` 要求未经过 torch.compile 的 CPU/eval 模型。
SAM2/EfficientTAM 加载器会显式关闭 config 中的 `compile_image_encoder`，避免与 TFLLM 路由叠加。

## 精度与覆盖

FP16/FP32 按外部输入输出 dtype 自动选择通道，外层参数类型保持原样；
FP32 输入与权重直接量化，I32 直接反量化到 FP32，不再逐层 FP32→FP16→FP32。
输出跟随输入或 CPU autocast；权重版本或 dtype 变化会重新打包。BF16 暂无直接通道，
已有 `precision="fp16"` 兼容入口仍可使用旧的 BF16→FP16 桥接。
NPU 使用现有 W8A8/I32 内核，FP16 通道保留既有舍入边界；真实任务精度仍需验证。
真实板端已复现 native 空间 Conv 的 INT32 错排；可创建
`NpuRuntime(spatial_convolution="gemm")` 并传给 `attach/load`，改用逐 patch 量化的
分块 im2col + NPU Linear 路径。该路径需要更新 Python 包，默认仍为 native；
SDK 已修复一个宽度分块的 INT32 连续写回缺陷，需要重编 `libNPU40T`。
2026-10-14 板端 native 重放 `[1,3,1024,1024]`、96 通道、3×3/stride=4/pad=1
连续三次与量化参考逐值一致，输出验证开启时热运行 7.11–7.31 ms。
stride=2、7×7 和整模型仍待复测，尚不能确认所有 native 错误已消除。
GEMM 的真实任务质量与端到端耗时也需单独验证。
DA3 官方显式 FP32 的相机编码/解码和 Gaussian 参数 head 保留在 Torch，避免将
姿态/FOV 等小型几何 head 与大视觉骨干一起量化。CPU 模式的空间卷积也使用现有
量化参考实现，因此 CPU 适配输出并非所有算子都与原始 FP32 模型逐字节相同。

标准 Linear、受支持 Conv2d 和 stride=2 的 ConvTranspose2d 被接管。官方 DINOv2 的简单 self-attention、SAM1 decoder 的
self/cross attention 使用专用 SDPA 适配；其它官方 SDPA 只在本 session 调用内路由。
相对位置的有限 additive bias、nested/custom mask、训练/dropout 或不满足融合条件的 attention
沿用原实现并记录 fallback。手写且未被识别的 attention 不算已融合。
超出原生限制的卷积/反卷积、depthwise、归一化、提示编码及插值仍可在 Torch 执行。
ConvTranspose2d 当前支持 groups=1、dilation=1、输入宽度≤512；完整限制见 TORCH.md。

`session.report.to_dict()` 区分静态 routed/skipped/retained 和实际 npu/npu_fused/cpu_reference/fallback。
默认 strict=false 保留必要的官方算子；strict=true 仅对已识别的请求路由拒绝不支持条件，
不能证明整个模型都在 NPU。设备/分配/执行错误始终抛出，不自动在 CPU 重跑。

## 定位端到端耗时

`backend="cpu"` 的 TFLLM Linear 是数值参考内核，其耗时不能用于估算 NPU 模式下
保留的 Torch CPU 算子耗时。NPU 模式仍需 CPU 激活量化、缓存同步、反量化；
FP32 模型的受支持 Linear/Conv/SDPA 直接读写 FP32；非连续张量可能按原 dtype 触发布局复制，
符合条件的窗口 SDPA 可直接读取通道连续的 FP32 strided 视图。
Torch 的 LayerNorm、GELU、插值、保留的卷积和 attention fallback 另有开销。

可直接包住现有调用，不需要重新加载模型或下载权重：

```python
import json
from tfllm.torch import InferenceProfiler

# 在此之前自行预热相同 shape，并用普通计时测量延迟。
# profiler 不自动重放调用，避免改变 predictor 的提示/视频状态。
with InferenceProfiler(trace_path="sam2_encode_trace.json") as profile:
    session.set_image(image)
print(profile)
with open("sam2_encode_profile.json", "w") as output:
    json.dump(profile.to_dict(), output, indent=2)
# 点击、SAM3 文字提示应分别包住 session.predict(...) 或 set_text_prompt(...)。
```

报告按输入 shape 列出算子的 self host time，并分组显示布局/类型转换、归一化、
激活、Torch 卷积、attention/matmul、插值和 TFLLM 调用。Chrome trace 可查看嵌套调用。
`tfllm::linear/attention/conv2d` 是同步调用的 host span，包含 CPU 工作和设备等待，
不是纯 NPU kernel 时间；native CPU worker 不会单独出现在 Torch profiler 中。
不要相加父子 total time，也不要把并发线程的 self time 总和视为端到端时间分解。
Profiler 有额外开销，最终速度仍用关闭 profiler 后的普通计时验收。

Torch 线程数独立于 TFDL 的 `cpuLimit`；此处 CLI 的 `--threads` 默认 4。
NPU 算子默认按工作量减少小画幅的分核，将 0–7 号 TFACC 视为相同算力，
对齐允许时均匀分配工作量；任务可由任意空闲核对执行。可用
`TFLLM_NPU_ADAPTIVE_SPLIT=0/1` 比较旧/新策略，`TFLLM_NPU_MAX_CORES=4` 将每个
执行组的计算核数上限设为 4（不固定物理核 ID）。门槛仍需板端校准；
测试脚本和作用范围见 [TORCH.md](TORCH.md#按工作量均匀分核)。
板端可分别用 4/8/16/32 个 Torch 线程测相同输入，同时记录实际 attention fallback。
TFLLM 的 CPU 池与 Torch 的线程池独立，增加线程数可能带来过度并行，应按实测选择。
通用量化/反量化内核可在启动前设置 `TFLLM_CPU_THREADS=16` 或 `32`（默认 8，范围 1..32）；
融合 attention 使用 `NpuRuntime(attention_fuse_workers=...)`，完整用法见 [TORCH.md](TORCH.md)。
Torch Linear 内部超过 1024 行时现已在一次调用内复用执行 workspace，减少连续分块的
缓冲区分配、命令重绑定和释放；复用范围结束时归还缓冲区，异常路径也会清理。
这项改动的板端收益需实测，不能用假 NPU 的延迟估计。

## 命令行与回归

```sh
python -m tfllm.vision list
python -m tfllm.vision segment --family sam --checkpoint /models/sam_vit_b.pth \
  --encoder vit_b --image frame.jpg --point 240 180 --output masks.npz --report coverage.json
python -m tfllm.vision depth --family depth-anything-v2 --checkpoint /models/depth_anything_v2_vits.pth \
  --encoder vits --image frame.jpg --output depth.npz --report coverage.json
# pip 安装后同样可使用 tfllm-vision；主机加 --backend cpu。
```

输出 NPZ 保留原始 mask/logits/scores 或浮点 depth/camera，不把可视化归一化当成数值结果。
测试由 `TFLLM_TORCH_TEST_PYTHON` 启用，使用 CPU 原生库和注入的假 NPU 库做路由/数值对照；
假设备测试不代表上板验收。真实 checkpoint 的精度、依赖版本兼容和 A77/NPU 性能需另行验证。
本轮本机使用 Torch 2.14.1、TorchVision 0.29.1、Transformers 5.19.0、NumPy 1.26.4：
SAM/SAM2/SAM3 小配置完整前向、官方 SAM decoder/DINOv2/EfficientTAM 编码器，
以及官方 EfficientTAM、DA2、DA3 完整随机权重模型已进行输出对照；
DA3 对照包括两个视图及相机结果。CMake 安装树、wheel 构建和安装后的 native 库发现已验证。

安装官方依赖后可运行完整模型对照（默认确定性随机权重，或传入本地 `--checkpoint`）：

```sh
python TFLLM/tests/vision_official_test.py --family efficienttam --backend cpu
python TFLLM/tests/vision_official_test.py --family depth-anything-v2 --backend cpu
python TFLLM/tests/vision_official_test.py --family depth-anything-v3 --backend cpu
```

它比较原模型与 TFLLM 的数值输出，包括 DA3 的多视图深度/置信度/相机参数，不能替代任务质量验证。
DA3 官方 API 导入还需要其相机/导出依赖（含 `addict`）；本机 macOS 如遇 Torch 与 pycolmap
各自携带 OpenMP 库冲突，须确保动态加载器使用同一份 OpenMP 库。

参考官方入口：[SAM](https://github.com/facebookresearch/segment-anything)、
[SAM2](https://github.com/facebookresearch/sam2)、[SAM3](https://github.com/facebookresearch/sam3)、
[EfficientTAM](https://github.com/yformer/EfficientTAM)、[Depth Anything V1](https://github.com/LiheYoung/Depth-Anything)、
[V2](https://github.com/DepthAnything/Depth-Anything-V2)、[V3](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)。


## 65 开发机的实测推荐入口（2026-10-17）

现有官方 checkpoint 可通过 `examples/vision_fast_inference.py` 使用按模型选择的配置。
完整分辨率，外部保留 FP32，Torch 40 线程；SAM2/SAM3 使用 NPU W8A8/I32 投影与
混合 attention，EfficientTAM-S 512 使用实测更快的原 Torch FP32。
`auto` 是固定配置选择，不会在设备执行失败后回放 CPU。
该入口包含独立的官方融合 MLP/大核 patch 卷积适配，无需导入外部 web demo。

```sh
# 65 上已安装；sam2-tiny / sam2-base / sam3 / efficienttam-s512 均可用。
python3 /root/TFDL2_SDK/bin/vision_fast_inference.py sam2-base --output /tmp/sam2.json
python3 /root/TFDL2_SDK/bin/vision_fast_inference.py sam3 --text cat --output /tmp/sam3.json
# 对照原 Torch；额外检查水平翻转图，模型加载/预热不进入计时。
python3 /root/TFDL2_SDK/bin/vision_fast_inference.py sam2-tiny --verify --flip-check --output /tmp/check.json
# 显式指定后端以复核。
python3 /root/TFDL2_SDK/bin/vision_fast_inference.py efficienttam-s512 --engine tfllm --output /tmp/eff-npu.json
```

默认图像是 SDK 的 `cat.jpg`；其他图像使用 `--image`，点提示使用 `--point X Y`。
JSON 旁保存 `.mask.npz`（最高分 `mask` 和对应 `score`）；序列化不进入推理计时。
CMake 安装树将两个脚本放在 `tfllm/test/`。默认预热 2 次、采样 5 次，可调整
`--warmup`/`--repeat`；加载一次模型后完成全部采样。

交互应用应常驻同一个 model/runtime：SAM2/EfficientTAM 每张新图只调用一次
`set_image`，随后多次 `predict`；SAM3 复用 `set_image` 返回的 `state` 再调用
`set_text_prompt`。每次调用 `segment` 会重新编码，适合独立图像请求。
分流参数、完整样本、负向实验和两种延迟口径见
[65 最快实测配置报告](artifacts/vision-fastest-20261017-server65/REPORT.md)。

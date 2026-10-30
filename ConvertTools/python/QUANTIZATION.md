# SDK 原生自动量化

转换入口仍是 `onnx2tfdl.py`，旧的 `quantContext()` 用法继续支持。
自动路径复用 SDK Python 的构图、校准、权重量化、执行和 FB 导出，不再依赖
`NPU40TQuantize/`、额外 FlatBuffer schema 或独立 C++ 辅助程序。

## 安装

自动流程使用 Python 3.10 或更新版本。需要在有 SDK 运行库的目标机器上重新构建 Python 绑定。本次增加了按名读取
TFContext 张量及量化配置的接口，并记录 Op 构图调用：

```sh
cd /root/TFDL2_SDK/Python
python3 -m pip install .
cd ../ConvertTools/python
python3 -m pip install -r requirements.txt
```

`sdk-cpu` 校准不依赖 Torch。`torch-cpu`、`torch-cuda` 和 `tfllm-npu` 另需安装对应 Torch/tfllm；
`torch-cuda` 需要支持 CUDA 的 PyTorch 及可用的 NVIDIA GPU，设备不可用时直接报错。
tfllm 需提供 `tfllm.torch.NpuRuntime`、`optimize`、`optimization_report`。
运行时统一使用已安装的 SDK Python 包及其配套库，不再通过 `--sdk-lib` 混用另一套库。

## 命令

```sh
python3 onnx2tfdl.py \
  --onnxpath /absolute/model.onnx --output_name /absolute/result/model \
  --std 255 255 255 --mean 0 0 0 --cvtype 0 \
  --quantimgDir /absolute/flower_photos --calibration-count 20 \
  --calibration-backend tfllm-npu \
  --auto-fallback --cosine-distance 0.002 --min-cosine 0.90 \
  --max-search-evals 128 --quant-workdir /absolute/result/work
```

`work` 必须为空。最终模型为 `model.quant.fb`。
`--calibration-backend` 可选 `sdk-cpu`、`torch-cpu`、`torch-cuda`、`tfllm-npu`。
校准数据三选一：图片目录、`--calibration-list` 图片路径列表，或
`--calibration-bin` 已预处理的小端 float32 `[N,C,H,W]` 文件。bin 不会再次除以 255。
图片按输入尺寸 INTER_LINEAR resize，`cvtype=0/1/2` 分别为 BGR/RGB/灰度，
遵循原转换器的 `(pixel-mean)/std`。目录分组均衡抽样，固定种子可用 `--seed` 设置。

当前图像流程要求单个固定 FP32 NCHW 输入、batch1、C=1 或 3。
`--fp16` 不与这条 FP32 参考守卫流程混用；回退恢复原始 FP32 权重和激活精度，
支持的浮点算子仍可由运行时动态量化送 NPU。

## 后端与自动回退

省略 `--auto-fallback` 时，仅用所选后端采集 MinMax，
把范围写回 SDK，然后直接调用原来的 `TFCalibration.Quantize` 导出，
不重建混合图、不运行候选搜索、不添加自动浮点回退。Python 手动指定的
`avoidtensors`、`stopquanttensors` 仍交给原生 Quantize 处理。

开启 `--auto-fallback` 后，`--calibration-backend` 同时决定范围采集、FP32 参考输出和
每轮候选评估的后端：`sdk-cpu` 使用 SDK CPU TFExecutor；`torch-cpu` 使用 Torch CPU；
`torch-cuda` 使用 `--cuda-device` 指定的 CUDA 设备；`tfllm-npu` 使用 `--chip` 指定的
tfllm NPU runtime。后端不可用时直接报错，不自动切换到 SDK CPU。

非 SDK 候选评估复用同一个 Torch 图和 runtime，按每轮浮点算子集合切换原始 FP32 权重与
SDK 量化权重的反量化值，并根据 SDK 激活范围模拟 Quantize/DeQuantize（包括浮点到量化
算子的输入边界）。这仍是混合精度模拟，不能保证与 SDK 整数内核逐位一致；报告中
`evaluator.approximate=true`。tfllm 的 Conv/Linear 加速沿用收集阶段的算子覆盖规则，
不支持的算子保留 Torch 执行，并记录实际 NPU 调用；完全没有 NPU 调用时失败。

非 SDK 模拟的 `mean_ms` 不代表 SDK FB 的推理延迟，所以达标配置优先选择更少的浮点
算子，再比较后端耗时；`sdk-cpu` 仍优先比较实测耗时。`report.json` 的
`search_objective` 记录该选择规则，`eval-*.json` 的 `backend` 记录候选执行后端。
`--calibration-workers` 和 `--calibration-compile` 目前只用于范围采集，候选评估仍串行。

最终导出仍由 SDK 构图，并重新加载实际 FB 做验收：`sdk-cpu` 在 SDK CPU 上验收，
其余后端在 SDK NPU 上验收（配置记录在 `run.json`，实际耗时记录在 `export_check`）。
独立验证集的 FP32 参考使用所选后端，候选搜索不使用该验证集。最终 FB 的余弦阈值作为
对照目标：未达到时记录 `satisfied=false` 并打印警告，保留模型并继续转换，不再次回退。
NaN/Inf、输出结构或类型异常、整数输出不一致和运行错误仍会报错。

```sh
# CUDA 校准，原生 SDK 量化导出，无自动回退
python3 onnx2tfdl.py \
  --onnxpath /absolute/model.onnx --output_name /absolute/result/cuda \
  --std 255 255 255 --mean 0 0 0 --cvtype 0 \
  --quantimgDir /absolute/flower_photos --calibration-count 20 \
  --calibration-backend torch-cuda --cuda-device 0 \
  --quant-workdir /absolute/result/cuda-work

# 同一路径改用 tfllm NPU 校准
python3 onnx2tfdl.py \
  --onnxpath /absolute/model.onnx --output_name /absolute/result/npu \
  --std 255 255 255 --mean 0 0 0 --cvtype 0 \
  --quantimgDir /absolute/flower_photos --calibration-count 20 \
  --calibration-backend tfllm-npu --chip 0 \
  --quant-workdir /absolute/result/npu-work
```

Python 入口同样支持：

```python
convertor.quantContext(
    calibration_list=images,
    decoderflags=DECODER_FLAGS.TFCV_BGR,
    calibration_backend="torch-cuda",  # 或 "tfllm-npu"
    auto_fallback=False,
    calibration_options={"cuda_device": 0, "threads": 8},
)
```

`--cuda-device` 是 `CUDA_VISIBLE_DEVICES` 过滤后的逻辑 GPU 编号，
`--chip` 是 tfllm 校准或候选/验收推理使用的 NPU 编号，二者独立。
CUDA 模式把参数、常量、输入和范围统计放到 GPU；关闭 autocast 和 TF32，
按 FP32 运算采集范围（图中显式 Cast 除外），结束后恢复调用者设置。
CPU/CUDA 内核仍可能存在浮点舍入差异，不承诺逐位一致。
CUDA/NPU 后端加速统计和自动回退评估，权重量化和 FB 导出仍由 SDK 完成。
只做校准导出时 CUDA 后端不执行 NPU 推理；开启回退时在 CUDA 上搜索，最终 FB 验收使用 SDK NPU。
未指定新后端/新校准选项时，旧的默认 SDK CPU 流程保持原样。
CUDA 精度控制按 [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
区分新旧接口，不混用 `fp32_precision` 与 `allow_tf32`。

## NPU 并行收集

`tfllm-npu` 新增 `--calibration-workers`（默认 1，范围 1–16）。
NPU 校准默认使用 native 卷积和单进程请求线程，无需额外选择卷积路径或执行器。
`--calibration-threads` 是进程内共享的 CPU/Torch 线程数；所有请求线程使用 `--chip`
指定的同一 NPU。样本按索引分片，各请求线程共享只读 Torch 图、权重和 NPU runtime，
分别收集统计，最后逐张量合并各分片的最小值和最大值，再执行原生量化。
当前实现针对 MinMax 收集；stop/avoid、自动回退和最终 SDK Quantize 的规则保持不变。

```sh
# 数据较多时：4 个请求线程，共享 16 个 Torch CPU 线程
python3 onnx2tfdl.py \
  --onnxpath /absolute/model.onnx --output_name /absolute/result/model \
  --std 255 255 255 --mean 0 0 0 --cvtype 0 \
  --quantimgDir /absolute/images --calibration-count 128 \
  --calibration-backend tfllm-npu --calibration-workers 4 --calibration-threads 16 \
  --quant-workdir /absolute/new-work
```

YOLO 例子也支持这两个参数。Python API 使用
`calibration_options={"workers": 4, "threads": 16, "chip": 0}`。
多 worker 目前仅支持 `tfllm-npu`；CPU/CUDA 仍用单 worker。
样本数少于请求实例数时，只启动非空分片，并在报告记录实际数量。
任何 worker 失败都会中止收集并停止其余实例，完整统计返回前不会执行 SDK Quantize。

以下为此前显式使用 gemm 卷积、独立进程执行器的历史记录，不代表当前默认配置：
12.65 实测：YOLOv8s、640×640、同一 NPU、40 个 CPU 核心。
所有配置的逐张量范围与串行结果完全一致：

| 实例数 × 每实例线程 | 20 张图 | 128 张图 |
|---|---:|---:|
| 1 × 8（原默认配置） | 8.34 s | 45.85 s |
| 1 × 16 | 6.66 s | 41.03 s |
| 4 × 4 | 6.67 s | 21.49 s |
| 4 × 8 | 6.47 s | 20.09 s |

上述历史进程模式中，各实例具有独立的线程配置、权重副本和 NPU 缓存占用。
当前默认请求线程共享模型和全局 Torch 线程配置。
三张图的抽样分段计时中，总收集 1.25 秒，范围统计约 0.10 秒，其他 Torch/Python
计算约 0.59 秒，原生算子 API 约 0.39 秒，准备与关闭约 0.17 秒。
原生 API 计时同时包含其 CPU 预处理、NPU 执行和回写。多个实例可交错处理这些工作。
这些数值包含模型准备、输入/图快照、进程启动、合并及 runtime 关闭，
不含图片预处理和 SDK Quantize。新报告的 `collector.seconds` 使用这个完整收集口径，
单实例另保留 `execution_seconds`，对应此前较窄的执行阶段。
校准入口已写出的输入文件会直接复用，避免为各 worker 重复预处理和复制整个数据集。

检测、姿态、分割（含 FP16 输出）和 SE-Inception 四类模型追加验证中，
串行/并行范围完全一致，导出后在两张独立图片上的 NPU 输出逐位一致。
这验证并行化保持统计与导出行为；各模型的原有量化误差仍需按任务数据集验收。
报告、分片索引和日志保留在 `work/collector-workers/` 与 `work/report.json`。
实测记录见 `tests/calibration-parallel-validation-20261028.json`。

### 单进程请求线程

NPU 校准默认使用一个只读 Torch 模型和一个 NPU runtime，
每个 worker 是一个请求线程；`--calibration-threads 16` 在主线程设置一次，
不会在各请求之间反复修改 Torch 的全局线程配置。
每个线程保留自己的样本分片与逐层统计，全部请求完成后合并 MinMax。
任一线程失败会停止剩余样本，并等待在途原生调用结束后关闭 runtime。
卷积默认走 native；命令行不再提供卷积路径与执行器选择参数。

```sh
export TFLLM_NPU_WORKSPACE_CACHE_INSTANCES=512
export TFLLM_NPU_WORKSPACE_CACHE_MIB=2048
python3 onnx2tfdl.py \
  --onnxpath /absolute/model.onnx --output_name /absolute/result/model \
  --std 255 255 255 --mean 0 0 0 --cvtype 0 \
  --quantimgDir /absolute/images --calibration-count 128 \
  --calibration-backend tfllm-npu \
  --calibration-workers 4 --calibration-threads 16 \
  --quant-workdir /absolute/new-work
```

`chip=0` 已调度第一颗芯片的 TFACC 0–7 八个核心，不是只用一个核心。
请求数和硬件核心数不必相等；四个 pair worker 与 CPU 工作的交错情况决定最佳实例数。
额外提供 `--calibration-compile`，通过配套 tfllm 的 `compile_snapshot()` 编译前向及
逐层范围统计。它仅支持线程执行器，要求固定形状和只读模型，不支持输入数据相关的
Python 分支。首次编译需要数十秒，完整 `collector.seconds` 会计入，不适合据此宣传小样本加速。

Python 配置为 `calibration_options={"workers":4,"threads":16,"compile":False}`。
`auto_fallback=False` 时仍直接使用原生 Quantize。
编译与线程模式均不修改 stop/avoid 和检测头量化边界。

此前 gemm 使用逐位置展开的近似卷积路径，与当前默认 native 的统计数值可能不同；
历史 gemm 结果应单独标注。线程数变化则应保持同一路径的 MinMax 一致。
CPU 编译融合也可能通过舍入改变量化边界，报告保留该配置，默认不启用。

2026-10-29 在 12.65 追加实测（YOLOv8s，Torch 16 线程，20 张相同校准图）：
native 路径完整收集从串行的 3.39 秒降到 4 个请求线程的 1.65 秒，8 个请求线程为
1.77 秒；三组逐张量 MinMax 完全一致。这组时间包含准备、收集合并与关闭，
不含图片预处理和 SDK Quantize。不要与上一节默认 gemm 路径的旧数据直接归因为线程加速。

另用 108 张独立 COCO128 图片测试固定形状的 `compile_snapshot()` 前向：

| 后端 | CPU intra-op 线程 | 请求线程 | 单请求平均耗时 | 总吞吐 |
|---|---:|---:|---:|---:|
| ORT CPU | 40 | 1 | 72.00 ms | 13.67 张/s |
| tfllm native + CPU 编译 | 16 | 1 | 60.67 ms | 16.19 张/s |
| ORT CPU | 16 | 8 | 357.91 ms | 21.77 张/s |
| tfllm native + CPU 编译 | 16 | 4 | 157.56 ms | 24.95 张/s |
| tfllm native + CPU 编译 | 16 | 8 | 326.40 ms | 23.79 张/s |
| ORT CPU | 32 | 8 | 224.51 ms | 34.55 张/s |

每组两轮；单请求耗时计同步前向，总吞吐还包含输出复制和请求调度。
输入已预处理，不计 resize、NMS、准备和编译。
tfllm 原生 runtime 还有自己的工作线程，表中 intra-op 数不表示进程线程总数。
单请求延迟已经优于所测 ORT 配置；最大并发吞吐仍未超过调优后的 ORT。
网络快照的本次首次调用含编译耗时 8.37 秒，已命中磁盘编译缓存；逐层范围快照另测
首次编译约 40.93 秒，20 张完整收集共 46.39 秒，因此少量校准图不要打开编译选项。

这 108 张图的 mAP50:95：ORT 54.14%，tfllm native 编译前向 53.23%（差 0.90 个百分点），
20 张 native 范围校准后导出的 quant.fb 为 50.91%（差 3.23 个百分点）。
后者不能据此认定量化精度达标。全部输出的 cosine 易被框坐标主导，分类分量的
平均 cosine 分别为 0.99515 和 0.96643，仍需检查任务指标。
这不是完整 COCO val2017 结果。新配置和精度边界、完整计时、测试及哈希记录见
`tests/tfllm-torch-speed-20261029.json`；原生 Quantize、stop/avoid 策略保持原样。

### 与 ONNX Runtime 的速度对照

以下保留此前默认 gemm 路径的定位数据。默认 worker 是独立进程，每个进程里的 Torch CPU 算子可使用多个线程；
`--calibration-threads` 控制 Torch intra-op 线程及子进程的 OMP/BLAS 配置，
并不限制整个进程的 OS 线程总数。tfllm 原生 runtime 还有自己的工作线程。
例如这次 8 线程配置的单实例进程观测到 56 个 OS 线程，其中包含辅助/等待线程。

当前 `tfllm-npu` 是逐算子的 Torch/NPU 混合校准。默认 `gemm` 卷积在 CPU 上
执行 `unfold` 和布局整理，再分块调用 NPU；输入动态量化、同步等待及输出转换也有开销。
它与最终编译好的 TFDL INT8 整图推理是不同流程，选择 NPU 后端不保证收集比 ORT 快。

12.65 上同一 YOLOv8s、同一 20 张已预处理的 640×640 图片，预热后各测两轮取平均：

| 稳态执行阶段（20 张） | 配置 8 线程 | 配置 16 线程 |
|---|---:|---:|
| ORT CPU，仅最终输出 | 2.74 s | 1.60 s |
| ORT CPU，逐层输出加 Min/Max | 4.00 s | 2.86 s |
| tfllm NPU，单实例逐层 Min/Max | 7.33 s | 6.47 s |

此表不含导入、图片预处理、模型/runtime 准备和关闭，也不含 SDK Quantize，
因此与上表包含完整收集启动/关闭的 `collector.seconds` 口径不同。
ORT 统计全部 333 个 ONNX 节点输出，Torch 统计 277 个 SDK 构图输出；两边节点边界不同，
该对照用于定位执行开销，不表示统计范围或校准数值可以直接互换。

8 线程 NPU 收集的 7.33 秒中，Min/Max 约 0.75 秒，CPU `unfold` 约 1.18 秒，
原生 Linear/Pointwise 接口约 2.33 秒，其余 padding、Torch 运算、布局整理及 Python
调度约 3.07 秒。原生接口时间包含 CPU 工作，并非纯 NPU 计时。
每张图有约 336 次原生 Linear/Pointwise 调用；关闭观察器后，前向仍约 6.19 秒/20 张。
多实例的加速来自交错处理这些工作；此前的 2.28 倍指相对 NPU 单实例收集的提升。
后续优化重点是减少 CPU 展开/复制、分块提交次数和逐算子精度边界转换。
完整记录见 `tests/calibration-ort-comparison-20261028.json`。

## 如何复用 SDK

- `Op.capture_operations()` 记录原转换器实际调用的算子，以及 TFSymbol 运算符，
  不另写 ONNX 导入器。候选模型通过这些 Op 调用重建。
- `TFCalibration` 完成 CPU 统计；tfllm 模式由同一份构图记录建立 Torch 观察器，
  Conv/Linear 送 NPU，其他运算和统计在 CPU。实际调用及 fallback 写入报告；零 NPU 调用报错。
- `TFContext.AddInt8Config` 写入统计；不开回退时直接 `TFCalibration.Quantize`。
  只有自动回退路径通过 `TFCalibration.QuantizeLite` 产生权重的
  SDK 量化结果；按消费者恢复浮点/量化权重，共享参数不会被另一分支覆盖。
- `Op.Quantize/DeQuantize` 明确处理分支精度边界，`TFContext.Modify` 设置输出类型。
  标量仿射计算复用 `Scale2`；需要浮点常量输入的二元计算保留 SDK 浮点内核并显式 Q/DQ。
  回退单元表示激活精度选择，不等于整个模型只有这些算子使用浮点计算。
- `TFExecutor` 运行 SDK CPU 候选及正式导出模型；非 SDK 候选使用对应后端的 Q/DQ 模拟。
  SDK 执行时 `TFTensor.toNumpy()` 解码全部输出，
  `TFContext.Dump` 编码 FB。没有 Python 手写 FB 编码步骤。

TFContext 的形状保持/广播等量化参数契约继续执行。自动回退路径的观察范围先在对应组内合并，
Sigmoid/Softmax 等理论范围保留 SDK 规则。自动回退中未验证的 UINT8 算子保留浮点并写入报告；
带隐式可训练权重的 Op 需改用 `Convolution2/InnerProduct2/Scale2` 等显式权重接口。
自定义算子或原生准备步骤产生未记录张量时明确报错。

tfllm 校准是近似动态量化计算，报告标记 `approximate_calibration`。
Torch 观察器目前覆盖常见 CNN、形状和二元运算；不支持的 Op/属性在打开设备前报错。
SDK CPU 后端不受这份 Torch 算子映射限制。

## 搜索和验收

先测 INT8 激活基线；未达标时测完整浮点回退守卫，再按拓扑分组尝试重新量化。
失败的组撤回后二分，广度优先继续；每次测当前完整组合，不假定层误差可相加。
SDK CPU 在预算内保留实测延迟最低的达标组合；非 SDK 模拟优先保留浮点算子最少的达标组合。
预算耗尽明确记录，不承诺全局最优。SDK CPU 每轮从原始构图和原始权重重建；
非 SDK 复用 Torch 图，并在原始权重和 SDK 量化权重的反量化值之间切换，避免累计量化误差。

`--tolerate` 默认 16：连续 16 个新的候选都未减少达标配置中的浮点算子数时提前退出，
并导出已有的最佳达标配置。`--tolerate 0` 关闭该提前退出条件，继续受
`--max-search-evals` 限制。缓存命中的重复配置不计数；基线和浮点守卫不计入停滞次数。
任一达标配置减少浮点算子数后计数清零，单纯耗时变小不清零。
这是启发式提前退出，可能错过队列中后续的改进。

每轮日志的 `float_units` 为本轮尝试值，`best_float_units` 为目前达标配置的最少浮点数，
`stalled_evaluations` 为连续停滞次数。最终 `search_stopped` 和报告中的 `search.stop_reason`
记录停止原因；因停滞退出时为 `tolerate`，不表示本轮候选已经达标。
最终 FB 检查保持独立验收，不会再次搜索或自动修改已导出的模型。
检查会打印并保存 `export_check`：实测 `mean_distance`、`mean_cosine`、`min_cosine`，
目标 `target_distance`、`target_min_cosine`，以及超出目标的 `distance_excess` 和
`min_cosine_shortfall`（未超出时为 0）。有搜索结果时还记录相对最佳模拟候选的差值：
`mean_distance_delta_from_search` 为实测距离减模拟距离，正值表示误差增大；
`min_cosine_delta_from_search` 为实测最低相似度减模拟最低相似度，负值表示变差。
余弦未达标时状态为 `exported_target_not_met`，该状态不代表模型导出失败。
实际任务精度仍应结合独立验证数据判断；显式设置的验证集硬门槛继续生效。

默认平均 cosine 距离 `1-cos <= 0.002`，所有样本/输出中的最低 cosine `>=0.90`。
多个浮点输出等权计入；整数输出精确比较。参考输出使用所选后端和原始浮点权重，
其中 tfllm 内部仍有动态量化误差。原 ONNX 转换可另用 `--check` 核验。

独立验收数据不会用于挑选回退组合：

```sh
# 在转换命令后添加
  --validation-bin /absolute/validation.f32 \
  --validation-labels /absolute/labels.json \
  --max-accuracy-loss 1.0
```

labels 为整数列表，或 `{"samples":[{"label_alphabetical":0}, ...]}`。
`--max-accuracy-loss 1.0` 是硬门槛，表示准确率损失不超过 1 个绝对百分点，并要求
验收样本不与校准张量重复。`--require-validation-target` 另外强制验收 cosine 达标。
未启用硬门槛时仍记录未达标状态，不将校准通过解释成验收通过。

普通量化输出 `source-float.fb`、`model.quant.fb`、`run.json`、`ranges.json`、`report.json`，
报告的 `quantization_mode` 为 `sdk-standard`，不代表已通过 cosine 验收。
自动回退另外生成 `profile.json`、逐候选 JSON、`reference.npz` 和导出检查 NPZ，
`quantization_mode` 为 `mixed-search`；显式验收另有验收 NPZ。
运行混合模型时使用同次生成的 `run.json`。

## 代码与测试

实现集中在 `TFConvertor/quantization/`：原生图重建、Torch 后端、范围收集、
`evaluation.py` 后端分派、`torch_evaluator.py` 混合精度模拟、搜索、指标及流程编排。
`calibration_backend.py` 只是转换器接入点。

```sh
python3 -m unittest discover -s tests -p 'test*.py' -v
```

`test_backend_evaluation.py` 使用真实 Torch CPU 和 SDK/tfllm stub 验证后端分派、混合精度
切换、搜索选择及 FB 验收流程；需要 NumPy/PyTorch，不需要原生 SDK 或 NPU。
CUDA 实机测试在没有 CUDA 时跳过，tfllm 实机精度和性能仍需目标机器验证。
纯策略测试不需 NPU；原生图测试需重建的 SDK Python 和运行库。
以下为候选评估改为跟随所选后端之前的历史记录，不代表当前模拟搜索的实机结果。
2026-10-28 在 12.65 的 20 张 flower / 16 候选集成检查中，tfllm 3540 次 NPU 调用，
正式导出平均距离约 0.00137。独立 500 张平均距离约 0.00145、最低 cosine 约 0.938，
准确率 96.6%，SDK FP32 参考 96.4%，延迟约 102 ms。逐层日志确认硬件 MatMul 执行。
这是有限预算的流程验证；默认 128 候选会改变组合，需重新验收。
这些 500 张已参与此前方案检查，不能视为未接触的最终测试集。

后端解耦验证见 `tests/backend-validation-20261028.json`：12.65 共 16 项通过、
1 项 CUDA 实机测试跳过；验证了标准量化、自动回退、显式验收和 CUDA 不可用报错。
SE 的 20 张未回退 MinMax 导出能在 NPU 运行，SDK CPU 校准的平均 cosine 约 0.519，
tfllm 校准约 0.457，均未达精度目标。这是后端/导出功能验证；
开启加速校准本身不会解决该模型的 MinMax 量化精度损失。CUDA 精度/性能仍需 GPU 实测。

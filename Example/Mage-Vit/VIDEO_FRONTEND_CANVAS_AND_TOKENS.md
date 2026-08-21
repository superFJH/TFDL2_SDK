# Mage-Vit 视频前端：从压缩视频到 Canvas 与视觉 Token

本文描述 `Example/Mage-Vit` **当前实际部署代码**中的完整视觉前端，解释一个
H.264/HEVC 视频如何经过 FFmpeg、块级信息筛选、Canvas 拼接、Mage-ViT 和
spatial merger，最终变成送入 Qwen3 的 2560 维视觉 Token。

本文对应的默认部署配置是：

```text
target_canvases    = 4
canvas             = 512 x 288 RGB
patch_size         = 16
spatial_merge_size = 2
vision encoder     = Mage-ViT 24 layers, hidden=1024
visual output      = 4 x 144 x 2560 = 576 x 2560
Qwen prefill       = fixed S=1024, right padding
```

主要实现文件：

- [`src/ffmpeg_decoder.cpp`](src/ffmpeg_decoder.cpp)：FFmpeg 扫描、解码、
  采样、时间戳和编码侧评分。
- [`src/codec_frontend.cpp`](src/codec_frontend.cpp)：候选块生成、Top-K、
  Canvas 拼接及原始位置记录。
- [`src/prompt_assembler.cpp`](src/prompt_assembler.cpp)：把视觉 Token 组织成
  Mage-VL 多模态文本片段。
- [`python/export_mage_embeddings.py`](python/export_mage_embeddings.py)：部署路径
  中的 Canvas 输入准备和视觉特征合并。
- [`python/build_mage_vit.py`](python/build_mage_vit.py)：Mage-ViT NPU 图结构、
  3D RoPE、Transformer 和 merger。
- [`deploy/persistent_runtime.py`](deploy/persistent_runtime.py)：Flask 服务中的
  4 路视觉 Executor 和 Qwen prompt 拼装。

## 1. 总体数据流

```text
H.264 / HEVC 文件
    │
    ▼
FFmpeg demux + 完整码流扫描/解码
    │  全视频均匀保留 32 个 DecodedFrame
    │  每帧含 RGB、frame index、PTS、key-frame 标志、codec score grid
    ▼
32 帧 × 若干 32x32 候选块
    │  bitcost / motion-vector 评分
    │  关键帧加分、单帧贡献限制
    ▼
Top-K 选出 576 个 32x32 block
    │  重新按原视频的 frame → y → x 排序
    ▼
4 个 512x288 Canvas
    │  每个 Canvas = 16x9 = 144 个 32x32 block
    │  同时记录每个 16x16 patch 的原始 (t,h,w)
    ▼
每个 Canvas 独立运行 Mage-ViT
    │  RGB [1,3,288,512]
    │  patch Conv → 576 x 1024
    │  24 层视觉 Transformer
    │  2x2 merger → 144 x 2560
    ▼
按 Canvas 序号拼接
    │
    ▼
visual_embeddings = [1,576,2560]
    │  scatter 到 576 个 <|image_pad|> 位置
    ▼
Qwen3 prompt，右侧补齐到 S=1024
```

Canvas 不是某一帧的缩放图，也不是视频的普通缩略图。它是一张由多个时间点、
多个空间位置的高价值 `32x32` 原始图像块拼成的马赛克图。

## 2. 固定预算如何计算

前端的核心配置在 `CodecFrontendConfig`：

```text
patch_size            P = 16
spatial_merge_size    M = 2
selection block       B = P × M = 32 pixels
canvas width          W = 512
canvas height         H = 288
group size            G = 32 frames
images per group      I = 4 canvases
deployed canvas count C = 4 canvases
```

每个 Canvas 的块数为：

```text
blocks_per_canvas
  = (W / B) × (H / B)
  = (512 / 32) × (288 / 32)
  = 16 × 9
  = 144
```

每个 `32x32` 块正好包含一个 `2x2` 的 `16x16` patch 组，因此在 merger 后
对应一个视觉 Token。于是：

```text
tokens_per_canvas = 144
total_visual_tokens = C × 144 = 576
```

部署配置需要保留的帧数为：

```text
sampled_frame_budget = (C / I) × G
                     = (4 / 4) × 32
                     = 32 frames
```

如果使用类默认值 `target_canvases=32`，则预算会变成 256 帧，分成 8 组，
每组生成 4 个 Canvas。Flask 部署显式传入 4，因此当前只处理一组 32 帧。

## 3. FFmpeg 如何扫描和保留帧

### 3.1 解码器初始化

前端用 `libavformat` 找到最佳视频流，用 `libavcodec` 打开对应解码器，并设置：

```text
AV_CODEC_FLAG2_EXPORT_MVS
threads = auto
thread_type = slice
```

`EXPORT_MVS` 请求 FFmpeg 把标准运动矢量附加到 `AVFrame`，用于没有 patched
bitcost 时的近似评分。

### 3.2 已知总帧数时的均匀采样

若 `AVStream::nb_frames=N` 有效，保留 `K=min(32,N)` 帧。第 `i` 个目标帧为：

```text
target[i] = round(i × (N - 1) / (K - 1)),  i = 0 ... K-1
```

因此首帧、尾帧和中间时间范围都被覆盖。这里的均匀是按解码帧序号，而不是按
画面内容评分；内容评分发生在后面的块级 Top-K。

### 3.3 未知总帧数时的 reservoir sample

若容器没有可靠的总帧数，前端采用固定种子的 reservoir sampling，使最终保留
集合近似均匀分布在完整流上，而且相同输入可重复得到同一结果。

### 3.4 为什么仍然需要扫描整个视频

即使最终只保留 32 帧，H.264/HEVC 的 P/B 帧依赖参考帧，当前实现仍然逐包扫描
并把帧送入解码器。未命中的帧会尽早丢弃，不执行 RGB 保存；命中的帧才通过
`sws_scale` 转换为原尺寸 `RGB24`。

因此当前的“均匀保留 32 帧”主要节省 RGB 转换、内存和后处理，并不等于只解码
32 张独立图片。这也是长视频上 FFmpeg 阶段仍可能明显耗时的原因。

### 3.5 时间戳和帧编号

每个保留帧记录两套时间信息：

- `frame_index`：从 0 开始的实际解码顺序编号。
- `timestamp_seconds`：优先使用 `best_effort_timestamp × time_base`；没有
  PTS 时才由估计帧率和帧编号计算。

后面 Mage 3D RoPE 的 `t` 使用 `frame_index + 1`，而 prompt 中显示的
`<12.6 seconds>` 使用 `timestamp_seconds`。二者用途不同，不能互换。

## 4. 块级信息评分

每个保留帧按 `32x32` 网格切成候选块：

```text
block rows = floor(frame_height / 32)
block cols = floor(frame_width  / 32)
```

不能组成完整 `32x32` 块的右边缘和下边缘会被忽略，当前不会缩放或补边。

### 4.1 patched FFmpeg bitcost

首选路径是弱符号接口：

```cpp
int mage_ffmpeg_get_bitcost(
    const AVFrame* frame,
    MageFfmpegBitcostView* view);
```

若部署链接了对应 patched FFmpeg shim，前端会复制每帧 macroblock/CTU 的
bitcost 网格。对一个 `32x32` 候选块，取其覆盖的 score cells 的平均值：

```text
codec_score(block) = mean(score_grid cells overlapped by block)
```

压缩编码器通常会给运动、纹理、预测困难或新出现的区域分配更多比特，因此
bitcost 可以作为“视觉信息量/重建难度”的廉价代理。

### 4.2 upstream FFmpeg 运动矢量 fallback

没有 patched bitcost 时，解码器读取 `AV_FRAME_DATA_MOTION_VECTORS`。每个运动
矢量在其目标位置所在的 `16x16` cell 中累计：

```text
mv_score = sqrt(motion_x² + motion_y²) / motion_scale
```

然后一个 `32x32` 块仍然取覆盖 cells 的平均值。这个路径偏向运动明显的区域，
但它并不等价于 bitcost：低运动高纹理区域可能被低估，压缩模式、残差和码率
信息也没有完整反映。

当前 `MotionVectorScore` 即使没有 side data 也会建立一个合法的全零网格。因此
普通 upstream FFmpeg 路径在“没有运动矢量”时通常得到零分网格，而不是自动
进入像素 fallback。

### 4.3 像素域 proxy

候选块代码还提供无合法 codec grid 时的像素 proxy。它每隔 4 像素采样亮度：

```text
Y = (77R + 150G + 29B) >> 8

pixel_score
  = mean(
      abs(Y_current - Y_previous)
      + 0.25 × abs(Y_current - Y_right)
    )
```

第一项近似时间运动/残差，第二项近似空间边缘。它是开发兼容路径，不应当作
官方 codec bitcost 的数值替代。

### 4.4 关键帧保护

关键帧中的每个候选块增加：

```text
keyframe_score_bonus = 1,000,000
```

目的不是直接保留整张关键帧，而是让关键帧作为场景锚点在 Top-K 中有很高的
生存概率。硬容量和单帧贡献上限仍然有效。

## 5. Top-K 和时间覆盖

对于当前一组 32 帧，需要选出：

```text
requested = 4 canvases × 144 blocks = 576 blocks
```

### 5.1 第一次排序

所有候选块稳定排序：

```text
1. score 从高到低
2. frame_index 从小到大
3. source_y 从小到大
4. source_x 从小到大
```

后三项同时提供确定性的同分 tie-break。

### 5.2 单帧贡献限制

第一轮选择限制每个源帧最多贡献：

```text
per_frame_cap = ceil(requested × 0.25)
              = ceil(576 × 0.25)
              = 144 blocks
```

因此正常第一轮中，一帧最多贡献相当于一个 Canvas 的容量，避免最高分单帧吞掉
全部时序预算。如果受限后不足 576 块，第二轮会从未使用候选中无上限补齐，保证
固定输出容量优先。

### 5.3 恢复因果时间顺序

Top-K 得到的块不会按分数顺序送给模型，而是再次按：

```text
frame_index → source_y → source_x
```

排序。这样 Qwen 看到的视觉 Token 大体保持从早到晚的顺序；块的精确原始位置
则由 3D RoPE 提供。

## 6. 576 个块如何拼成 4 个 Canvas

排序后的 576 个块顺序切成 4 份，每份 144 个。每份按 Canvas 行优先位置复制：

```text
destination_x = (i % 16) × 32
destination_y = (i / 16) × 32
```

所以每个 Canvas 是 `16 × 9` 个 block 的规则网格。块像素直接从源 RGB 帧复制，
不做 JPEG 往返、不做块内缩放。

### 6.1 拼接位置不是原始位置

模型输入图上的 `(destination_x,destination_y)` 只负责把 RGB 放进固定 Tensor。
每个 block 还会保存它在原视频中的真实 patch 坐标。

假设 block 左上角为 `(source_x,source_y)`，则：

```text
t = frame_index + 1
h = source_y / 16
w = source_x / 16
```

一个 `32x32` block 包含四个 `16x16` patch，严格按 merger 需要的顺序记录：

```text
(t,h,  w)    # top-left
(t,h,  w+1)  # top-right
(t,h+1,w)    # bottom-left
(t,h+1,w+1)  # bottom-right
```

每个 Canvas 因而有：

```text
144 blocks × 4 positions = 576 patch_positions
```

四个 Canvas 总共有 2304 个 patch positions，但 merger 后仍是 576 个视觉
Token。每个位置还平行保存对应源帧的秒级时间戳。

Canvas 自身的 `timestamp_seconds` 是其 144 个源块时间戳的中位数，主要用于
描述和 fallback；精确 prompt span 使用每个 block 保存的时间戳。

## 7. 前端 bundle 的文件契约

原生前端写出：

```text
canvas_000.ppm
canvas_001.ppm
canvas_002.ppm
canvas_003.ppm
manifest.json
vision_content.txt
metrics.json
```

`manifest.json` 的核心结构为：

```json
{
  "patch_size": 16,
  "spatial_merge_size": 2,
  "canvas_width": 512,
  "canvas_height": 288,
  "canvases": [
    {
      "file": "canvas_000.ppm",
      "timestamp_seconds": 4.2,
      "token_count": 144,
      "patch_positions": [[1, 0, 0], [1, 0, 1]],
      "patch_timestamps": [0.0, 0.0]
    }
  ]
}
```

实际 `patch_positions` 和 `patch_timestamps` 每个 Canvas 都有 576 项。PPM
使用无损 RGB，避免 JPEG 再编码改变视觉编码器输入。

独立 C++ 模式在同一进程中可直接把内存 Canvas 送入 VisionRunner；当前 Flask
部署将 codec frontend 作为子进程，所以使用 PPM 和 manifest 作为 C++ → Python
进程边界。

## 8. Canvas 如何进入 Mage-ViT

### 8.1 三个图输入

每个 Canvas 单独运行一次相同的视觉 FB：

```text
input 0: raw RGB               [1,3,288,512]
input 1: Mage 3D RoPE sin      [1,1,576,64]
input 2: Mage 3D RoPE cos      [1,1,576,64]
output : merged embeddings     [1,144,2560]
```

RGB 从 PPM 的 HWC 转为 NCHW。NPU 图中的 Placeholder 编码了预处理：

```text
normalized[c] = (rgb[c] / 255 - mean[c]) / std[c]

mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
```

### 8.2 Patch embedding

Mage-ViT 使用 `kernel=16, stride=16` 的 patch convolution：

```text
[1,3,288,512]
  → [1,1024,18,32]
  → 18 × 32 = 576 patch tokens
  → [1,576,1024]
```

卷积天然产生普通行优先 patch 顺序。图随后将它重排成每个 `2x2` block 内的
`TL,TR,BL,BR` 顺序，使视觉特征顺序和 manifest 中的 `patch_positions` 完全一致。

### 8.3 Mage 4:6:6 三维 RoPE

视觉塔参数为：

```text
hidden_size = 1024
heads       = 16
head_dim    = 64
theta       = 10000
```

每个 patch 的位置为 `(t,h,w)`。RoPE 在半个 head 维度上按 `T:H:W = 4:6:6`
分配频率；`head_dim=64` 时即：

```text
half_dim = 32
T dims   = 8
H dims   = 12
W dims   = 12
```

三组频率拼接后复制到 64 维，生成 sin/cos，再以相邻偶奇维旋转 Q 和 K。这样
Transformer 处理的是固定 Canvas 图像，但注意力仍感知每个 patch 在原视频中的
时间、原始高度和原始宽度，而不是错误地把马赛克相邻块当成原图相邻区域。

### 8.4 24 层视觉 Transformer

每个 Canvas 的 576 patch tokens 独立通过 24 层视觉 Transformer：

```text
LayerNorm
QKV projection
3D RoPE(Q,K)
QK / sqrt(64)
Softmax
Attention × V
output projection + residual
LayerNorm
MLP(GELU) + residual
```

四个 Canvas 之间不做视觉塔内部 attention。官方非 FlashAttention 路径本身也
按 Canvas 切成独立注意力序列，因此“一次执行一个 Canvas”保留了该注意力边界，
同时避免为多个 Canvas 添加视觉 padding mask。

部署 FB 使用 INT8/FP16 hybrid：普通投影和 MatMul 走 INT8，LayerNorm、残差、
最终 merger 以及绝对范围最大的 Top-K=2 分支保留 FP16；QK 使用 H×S 逐行量化
参数。无论内部量化方式如何，交给语言模型的稳定 ABI 都是 2560 维浮点 embedding。

## 9. 为什么 576 patch tokens 最终只有 144 个视觉 Token

24 层 Transformer 后仍为：

```text
[1,576,1024]
```

spatial merger 将每个连续 `2x2` patch 组拼接：

```text
4 × 1024 = 4096 dimensions
```

然后执行：

```text
LayerNorm
Linear 4096 → 4096
GELU
Linear 4096 → 2560
```

因此：

```text
[1,576,1024]
  → group every 4 patches
  → [1,144,4096]
  → merger MLP
  → [1,144,2560]
```

这里的 2560 正好等于 Qwen3 的 hidden size，所以 merger 输出可以直接替换
Qwen embedding table 中的 `<|image_pad|>` embedding，无需额外维度投影。

## 10. 四路视觉编码与 Token 合并

Flask worker 启动时建立：

```text
1 个共享权重的 TFContext
4 个各自归属一个线程的 TFExecutor
```

四张 Canvas 并行执行。每个 Executor 不会被两个线程同时调用。完成后按照原始
Canvas index 排序，而不是按照线程完成先后排序：

```text
canvas 0 [1,144,2560]
canvas 1 [1,144,2560]
canvas 2 [1,144,2560]
canvas 3 [1,144,2560]
              │ concat token axis
              ▼
combined [1,576,2560]
```

NPU 输出边界当前为 FP16，Host 为后续 Python/ORT 兼容将其转换并保存为
`visual_embeddings.f32`。这里转换的是存储表示，不改变 Token 数量或顺序。

## 11. 视觉 Token 如何进入 Qwen prompt

前端遍历 merger 后的 block/token 顺序。相邻 Token 如果来自相同源帧且时间戳
相同，就合并成一个 span：

```text
<12.6 seconds><|vision_start|>
<|image_pad|><|image_pad|>...<|vision_end|>
```

每个 merged visual token 对应一个 `<|image_pad|>`。四个 Canvas 总数必须严格为：

```text
image_pad count = visual embedding rows = 576
```

Host 随后执行 Qwen chat template 和 tokenizer，读取词嵌入表，并把所有
`image_token_id` 位置替换为对应的 2560 维视觉 embedding。系统前缀、视觉 span、
用户问题和 generation prompt 共同组成真实序列。

当前 NPU prefill 的物理长度固定为 1024：

- 真实序列不足 1024 时在右侧补 pad。
- 有效 Token 的 causal attention 不会看到未来的 pad。
- 只把真实有效前缀的 K/V cache 导出给 CPU decoder。
- 真实组装长度超过 1024 时在 prefill 前报错，不会静默截断视频或问题。

## 12. 当前部署的同步边界

当前 Flask 路径的实际顺序是：

```text
1. Python 启动 megavit_frontend 子进程
2. 子进程扫描视频、选块、生成全部 4 个 PPM 和 manifest
3. 子进程退出
4. Python 读取完整 manifest
5. 4 个 Vision Executor 并行处理 4 张 Canvas
```

也就是说，现在实现的是“4 张 Canvas 的视觉 NPU 并行”，尚未实现“FFmpeg
生产一个 Canvas，NPU 立刻消费一个 Canvas”的跨进程流水。

在当前 4-Canvas 算法里，576 个块是在同一组 32 帧的全部候选上做全局 Top-K。
在看完这 32 帧并完成评分之前，无法严格确定第一张 Canvas 应该包含哪些块。
完成 Top-K 后，Canvas pack 本身通常只有毫秒级，因此让 pack 和 NPU 再流水的
收益很小。若要显著重叠解码与 NPU，需要改变选择算法，例如分小组提前定稿，
这会改变 Canvas 内容和精度特性，不只是工程调度修改。

## 13. 当前算法的设计原理

### 固定视觉预算

无论视频原始分辨率和时长如何，部署默认要么输出 4 个固定尺寸 Canvas 和 576
个视觉 Token，要么在无法填满固定预算时明确报错。这样视觉 NPU 图、Qwen
prefill 容量、KV cache 和延迟都可预测。

### 使用编码器信息作为显著性代理

视频编码器已经为运动、纹理和预测残差计算了大量信息。复用 bitcost 或运动矢量
比对每一帧再运行一个独立的显著性网络更便宜。

### 高信息区域与时间覆盖平衡

纯 Top-K 容易被单帧占满；`per_frame_cap_ratio` 强制第一轮保留时间多样性。
关键帧 bonus 又保证场景锚点不因运动较少而完全消失。

### 像素打包与语义位置解耦

RGB block 被紧凑放入固定 Canvas 只是为了高效运行常规 ViT patch Conv；真正的
原视频时空关系由 `(t,h,w)` 3D RoPE 恢复。因此 Canvas 可以是马赛克，同时不必
把马赛克几何误认为原视频几何。

### merger 降低语言模型成本

视觉 Transformer 先在 576 个细粒度 patch 上提取特征，再用 2x2 merger 压成
144 个语言侧 Token。这样保留视觉局部建模能力，同时将 Qwen attention 的视觉
序列长度降低 4 倍。

## 14. 边界条件和已知限制

1. `target_canvases` 必须为 `images_per_group=4` 的整数倍；当前部署固定为 4。
2. 一组少于 `min_group_frames=8` 时不会生成完整 Canvas，服务最终报错。
3. `require_full_canvases=true`；候选块不足以填满固定容量时直接报错，不输出
   部分 Canvas。
4. 源帧宽高不是 32 的倍数时，右边和下边不足一个 block 的像素被忽略。
5. upstream FFmpeg 运动矢量只是 bitcost 近似；`patched_bitcost=false` 时不应
   假设选择结果与微软 codec-video-prep 一致。
6. 当前 grouping/Top-K 是确定性的 compact readiness-style 近似，不是官方
   codec-video-prep 全部策略的逐行复刻。
7. 视觉 INT8 范围目前来自有限校准视频。更换视频在结构上无需重建 FB，但生产
   精度需要在多场景、不同码率、不同运动强度视频上重新聚合一次代表性校准集。
8. 更改 Canvas 尺寸需要重新导出视觉 FB；更改 Canvas 数量会改变视觉 Token 数、
   prompt 长度和 prefill 校准分布，部署上应视为另一个 profile。

## 15. 运行和检查

只生成 Canvas bundle，不运行视觉 NPU：

```bash
Example/Mage-Vit/build/megavit_frontend \
  --video sample.mp4 \
  --target-canvases 4 \
  --output-dir /tmp/megavit-canvas
```

检查摘要：

```bash
python3 -m json.tool /tmp/megavit-canvas/metrics.json
python3 -m json.tool /tmp/megavit-canvas/manifest.json
```

当前 soccer 部署样例的实际摘要为：

```json
{
  "decoded_frames": 32,
  "canvases": 4,
  "visual_tokens": 576,
  "decode_ms": 1718.43,
  "canvas_select_pack_ms": 8.38
}
```

这组数据也说明：该样例中瓶颈主要是完整视频扫描/解码，而不是 Top-K 和 Canvas
拼接。视觉 NPU 的耗时由 Flask 持久运行时单独记录为 `vision.execute_seconds`、
`vision.sum_canvas_execute_seconds` 和 `vision.per_canvas_seconds`。

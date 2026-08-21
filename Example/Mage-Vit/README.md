# Mage-Vit: Mage-VL codec/NPU visual frontend

> The directory name follows the requested `Example/Mage-Vit` spelling. The
> upstream model and all code-level identifiers use the correct **Mage-VL /
> Mage-ViT** name.

This example implements the low-risk hybrid deployment boundary:

```text
H.264 / HEVC
  -> FFmpeg C++ decode + codec score adapter
  -> readiness-style grouping / Top-K 2x2 patch blocks
  -> fixed RGB canvases + (t,h,w) patch positions
  -> mage_vit.quant.fb on TFDL/NPU
  -> [visual_tokens, 2560] float embeddings
  -> Host tokenizer / visual-token scatter
  -> Qwen3-4B prefill on TFDL/NPU
  -> Qwen3-4B W8A8 token decode on ONNX Runtime/CPU
```

视频如何从 FFmpeg 解码、块级 Top-K 变成固定 Canvas，并进一步生成
`[576,2560]` 视觉 Token，详见
[`VIDEO_FRONTEND_CANVAS_AND_TOKENS.md`](VIDEO_FRONTEND_CANVAS_AND_TOKENS.md)。

TFDL does not contain string processing or a tokenizer. The visual boundary is
the 2560-dimensional embeddings; the Qwen prefill boundary is a complete FP16
K/V bundle plus last-token logits. The reference CPU decode engine now uses
ONNX Runtime W8A8, while the earlier PyTorch CPU/GPU path remains available for
alignment.

## What is implemented

- Dependency-free codec selection/canvas core with deterministic tests.
- Optional C++ FFmpeg decoder. It scans the complete stream while retaining a
  bounded uniform frame sample.
- Stable weak-symbol shim for a patched-FFmpeg macroblock/CTU bitcost exporter.
- Upstream-FFmpeg fallback based on exported motion vectors plus a pixel-domain
  residual/edge proxy.
- In-memory 2x2 patch-block canvas packing and exact `(t,h,w)` metadata.
- Exact Mage 4:6:6 vision RoPE construction.
- Direct TFDL graph builder for the joint checkpoint's `model.visual.*`
  weights: patch Conv, pre-LN, 24 Transformer blocks, merger and 2560 output.
- Linear projections represented as Conv1x1; QK/AV remain activation MatMul.
- Float and INT8 dump path with external min/max calibration ranges.
- ViT.py-style source precision graph: FP16 LayerNorm/residual/merger, INT8
  Attention/MLP compute, and absolute-range Top-K FP16 branch bypass. The
  quantizer encodes this graph with QuantizeLite; it does not rewrite it.
- C++ TFDL runner that adapts raw RGB to the FB input dtype plus runtime
  sin/cos inputs.
- A Python Qwen3-only loader that does not allocate the Mage visual tower on
  the CPU/GPU.
- Stage-level timing JSON for hybrid-versus-GPU evaluation.
- A single-session ONNX Runtime decoder with per-channel W8 weights, dynamic
  U8 activations and direct import of the NPU-prefill FP16 KV bundle.
- A resumable one-command video -> NPU vision -> NPU prefill -> CPU decode
  runner in `run_ort_pipeline.py`.

The standalone `microsoft/Mage-ViT` checkpoint is intentionally not used. It is
only ViT-pretrained; this example extracts the jointly trained visual weights
from `microsoft/Mage-VL`.

## Deploy the Flask service

The checked deployment package under [`deploy/`](deploy/) contains the single
selected vision FB, all 36 fixed-S=1024 Qwen prefill FBs, the W8A8 ORT decoder
external data, addon library, integrity manifest and Flask service. Target
machine build/install/start instructions and the HTTP contract are in
[`deploy/README.md`](deploy/README.md). The current bucket accepts four
canvases and custom questions up to 120 tokenizer tokens by right-padding the
NPU graph and exporting only the valid KV prefix. A different canvas count
still requires another FB bucket.

The deployment root is an interactive browser demo: video upload, custom
question, streaming token output, per-stage progress, TTFT, decode duration
and ORT throughput. Blocking JSON and streaming NDJSON APIs are both retained.
The Flask worker owns a process-resident Vision Executor, all 36 prefill
Context/Executors and the ORT decoder session; requests reuse them rather than
reloading FBs or rebuilding sessions.

## Build and smoke-test the C++ frontend

From the SDK root:

```bash
cmake -S Example/Mage-Vit -B /tmp/megavit-build \
  -DMEGAVIT_WITH_FFMPEG=ON \
  -DMEGAVIT_WITH_TFDL=ON
cmake --build /tmp/megavit-build -j2
ctest --test-dir /tmp/megavit-build --output-on-failure
```

If FFmpeg development packages are absent, CMake builds a decoder stub and the
synthetic pipeline remains available:

```bash
/tmp/megavit-build/megavit_frontend \
  --synthetic \
  --output-dir /tmp/megavit-smoke
```

Outputs:

```text
manifest.json             canvas timestamps and block-ordered patch positions
canvas_*.ppm               inspection/calibration assets only
vision_content.txt         timestamp + image_pad multimodal content fragment
metrics.json               decode/select-pack/TFDL stage timings
visual_embeddings.f32      present when --model is supplied
```

PPM files are an evaluation artifact. The runtime path passes the RGB vector
directly to TFDL and performs no JPEG encode/decode.

## FFmpeg integration

The real decoder target requires `libavformat`, `libavcodec`, `libavutil` and
`libswscale` development packages discoverable with `pkg-config`.

Upstream FFmpeg does not export Mage's bitcost map. The example therefore
declares this stable adapter in
[`patched_ffmpeg_adapter.hpp`](include/megavit/patched_ffmpeg_adapter.hpp):

```cpp
extern "C" int mage_ffmpeg_get_bitcost(
    const AVFrame* frame,
    MageFfmpegBitcostView* view);
```

A small shim compiled against the selected codec-video-prep/patched-FFmpeg
version should implement that symbol. When it is linked, the decoder copies its
macroblock/CTU values. Without it, the same binary falls back to standard
`AV_FRAME_DATA_MOTION_VECTORS`; the canvas contract remains unchanged but model
quality is not expected to match the official codec path.

The current reference decoder performs one complete software decode and only
converts retained frames to RGB. A production optimization should use the
official two-pass strategy:

1. bitcost-only scan with IDCT/loop-filter skipped;
2. keyframe seek and pixel decode only around selected frames.

That optimization belongs entirely behind `VideoDecoder` and does not change
the canvas or NPU interfaces.

## Build `mage_vit.fb` / `mage_vit.quant.fb`

Activate the SDK Python environment and point `--model-path` at a downloaded
`microsoft/Mage-VL` snapshot:

```bash
source .venv-tfdl-linux/bin/activate

python Example/Mage-Vit/python/build_mage_vit.py \
  --model-path /models/Mage-VL \
  --canvas-size 288 512 \
  --dump-fb /tmp/mage_vit.fb \
  --dump-symbol-map /tmp/mage_vit.symbols.json
```

Generate real codec canvases before INT8 calibration:

```bash
/tmp/megavit-build/megavit_frontend \
  --video sample.mp4 \
  --output-dir /tmp/mage-calib

python Example/Mage-Vit/python/build_mage_vit.py \
  --model-path /models/Mage-VL \
  --canvas-size 288 512 \
  --bundle /tmp/mage-calib \
  --max-calib 32 \
  --dump-ranges /tmp/mage_vit.minmax.json \
  --dump-quant-fb /tmp/mage_vit.quant.fb \
  --quant-profile int8-fp16-topk \
  --outlier-top-k 2 \
  --dump-bypass-report /tmp/mage_vit.topk.json \
  --dump-symbol-map /tmp/mage_vit.symbols.json
```

`int8-fp16-topk` is the default profile. It ranks each block's Attention output
projection and MLP output by `max(abs(min), abs(max))` over the calibration
set. LayerNorm/residual streams and the final merger stay FP16; all unselected
branches remain INT8. A selected Attention candidate restores only its output
projection, while a selected MLP candidate restores its complete MLP. These
Q/DQ boundaries and FP16 parameters are constructed directly in the source
graph. QuantizeLite only encodes weights/qinfo; no post-quant Modify is used.

Quantized exports also enable `--per-channel-qk` by default. Calibration keeps
one range for every `(head, query)` QK row (`H*S` ranges). The source graph
contains an identity-code Requant whose output qinfo is the QK qinfo divided
by `sqrt(head_dim)`. Before registration, any row
whose `Q_scale*K_scale/QK_row_scale` is too large is expanded while preserving
its min/max ratio and zero point. The default maximum is `0.99`, because the
gemmlowp `QuantizeMultiplierSmallerThanOne` path requires a value strictly
below one. Use `--no-per-channel-qk` only for a scalar-QK A/B export.

Range JSON files generated by older revisions contain only scalar extrema and
cannot be used with this default. Regenerate them with `--dump-ranges`; the new
file contains `layers.N.qk_matmul.rows` and `layers.N.attn_scores.rows` arrays.

The reference graph has a fixed 288x512 landscape canvas profile:

- 18x32 = 576 vision patches;
- 2x2 merger gives 144 LLM tokens per canvas;
- 32 canvases give 4608 visual tokens.

One canvas is invoked at a time. This exactly preserves the official
per-canvas attention boundary and avoids padding masks. Additional fixed
profiles can be built by changing `--canvas-size`; both dimensions must be a
multiple of 32.

Compare a dumped graph with the checkpoint reference on an existing frontend
bundle:

```bash
python Example/Mage-Vit/python/evaluate_mage_vit.py \
  --model-path /models/Mage-VL \
  --bundle /tmp/mage-calib \
  --fb /tmp/mage_vit.quant.fb \
  --output-json /tmp/mage_vit.precision.json
```

Before the H×S QK update, the scalar-QK exporter produced these one-canvas
results on the bundled Microsoft soccer example processed by
codec-video-prep 0.2.5:

| Top-K | FP16 bypass | Size (MiB) | Cosine vs PyTorch |
|---:|---|---:|---:|
| 0 | none | 345.7 | 0.7377 |
| 1 | layer 14 MLP | 353.6 | 0.7953 |
| 2 | layer 14/23 MLP | 361.6 | 0.8102 |
| 4 | four MLPs | 377.4 | 0.8177 |
| 8 | seven MLPs + one Attention projection | 402.1 | 0.8217 |

Top-K=2 remains the default speed-oriented point. With the current H×S QK
export, four sampled canvases improved from scalar-QK mean cosine 0.8148 to
0.8283 (range 0.8217-0.8413). The earlier TFDL dtype audit found 877 UINT8,
167 FP16 and three FP32 tensors; FP32 is limited to the source raw-pixel input
and two RoPE inputs, and the 2560-wide output is FP16. See
[EVALUATION.md](EVALUATION.md) for the exact setup, per-canvas results and
limitations.

The complete 4608-token test video was also passed through the official Qwen
backend with either official BF16 vision features or Top-K=2 INT8/FP16 vision
features. Four directly verifiable factual questions scored 4/4 on both paths;
across eight questions, initial logits cosine averaged 0.9688 and initial
greedy tokens agreed 7/8. See
[QWEN_AB_EVALUATION.md](QWEN_AB_EVALUATION.md).

The Python smoke test validates block ordering/RoPE without TFDL. The extended
test also builds a small graph, checks float parity and executes the INT8 graph:

```bash
python Example/Mage-Vit/python/test_mage_vit.py
python Example/Mage-Vit/python/test_mage_vit.py --with-tfdl
```

## Run the NPU vision frontend

```bash
/tmp/megavit-build/megavit_frontend \
  --video sample.mp4 \
  --model /tmp/mage_vit.quant.fb \
  --addon AddonOps/build/libTFDLAddOn.so \
  --executor-config Example/Mage-Vit/runconfig.json \
  --output-dir /tmp/mage-run
```

The quantized graph consumes raw RGB bytes. Mean/std normalization is encoded
in the TFDL placeholder. RoPE sin/cos remain float inputs, and the default
final embedding boundary is FP16. The C++ runner converts it to float32 for the
host-side Qwen bridge.

## Run Qwen3 on CPU/GPU

The reference bridge loads only the text model weights from the Mage checkpoint
and performs greedy decoding:

```bash
python Example/Mage-Vit/python/qwen3_bridge.py \
  --model-path /models/Mage-VL \
  --bundle /tmp/mage-run \
  --device cuda \
  --dtype bfloat16 \
  --question "Describe this video."
```

Mage-VL requires `transformers>=5.7`. Keep the Qwen runtime in a separate
environment if the TFDL conversion environment pins an older Transformers
release. The bridge now rejects older versions instead of returning unreliable
multimodal output.

For a 1000-2000 RMB GPU, an engine with weight-only INT8/INT4 and efficient KV
cache is likely more useful than this PyTorch reference. The stable integration
contract is:

```text
input_ids + attention_mask + image_token offsets + float visual embeddings
```

so the Qwen backend can be replaced without rebuilding the NPU graph.

## Experimental NPU Qwen prefill

[`Qwen-prefill/README.md`](Qwen-prefill/README.md) contains the fixed-sequence
36-layer prefill exporter and runner. It keeps RMSNorm, RoPE, residual and KV
cache in FP16; QK/AV and normal projections are UINT8. QK uses H*S row qinfo,
the attention scale is folded into an identity Requant, and an ARM custom op
fuses causal masking with UINT8 Softmax. The original CPU/GPU Qwen bridge above
remains unchanged and is still the supported full-generation reference.

## Price/performance evaluation

Compare the same codec canvases and token budget in three configurations:

```text
A. FFmpeg CPU + GPU Mage-ViT + GPU Qwen3
B. FFmpeg CPU + NPU Mage-ViT + GPU Qwen3
C. FFmpeg CPU + NPU Mage-ViT + CPU Qwen3
```

Record warm batch-1 and concurrent-stream measurements:

- `decode_ms`, `canvas_select_pack_ms`, `tfdl_vision_ms` from `metrics.json`;
- Qwen `prefill_ms`, decode latency and tokens/s from `qwen3_bridge.py` or the
  selected optimized backend;
- incremental wall power and joules/query;
- maximum streams meeting the TTFT SLA;
- answer quality using identical canvases and greedy decoding.

Use SLA-qualified concurrency rather than peak TOPS:

```text
cost_per_stream = (hardware amortization + power) / concurrent_streams_at_SLA
```

The hybrid route is commercially compelling when the NPU is already part of
the CPU platform, when vision and Qwen can overlap across streams, or when the
offload permits a cheaper GPU tier. For a single serial request on a GPU that
must already host Qwen3, pure GPU can remain faster.

## Current limitations and acceptance gates

1. The adapter and fallback are implemented, but exact Microsoft bitcost data
   requires a shim for the chosen patched FFmpeg build.
2. The grouping/selection implementation is deterministic and preserves the
   Mage tensor contract, but it is a compact readiness-style approximation.
   Compare its selected positions against codec-video-prep before judging model
   accuracy.
3. The default `int8-fp16-topk --outlier-top-k 2` profile meets the requested
   cosine threshold on four canvases from one reference video. Recalibrate and
   repeat the sweep on a broader video set before freezing a production Top-K.
4. The Python Qwen bridge is a correctness/performance reference, not the final
   low-cost GPU engine.
5. Full-checkpoint float and INT8 conversion has been executed. The float graph
   reached cosine 0.99999999999 versus PyTorch. Hardware NPU latency remains an
   acceptance gate on this host because `/dev/thinkforce0` is unavailable and
   the driver reports `please insmod driver`.

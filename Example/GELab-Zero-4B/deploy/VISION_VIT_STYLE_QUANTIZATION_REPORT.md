# GELab Vision: Vit.py-style FP16 and W8A8 experiment

## Graph contract

The selected FP16 graph and experimental W8A8 graph are built with the direct
`TFContext` pattern from `ConvertTools/python/example/Vit.py`:

- main token stream: Conv-native BCS `[B,C,S]`;
- parameterized Linear lowering: Conv1x1;
- projection entry: balanced Reshape `[B,C,S] -> [B,C,H,W]`, without
  Transpose; 576→24x24, 144→12x12 and bucket 64→8x8;
- QK and AV: MatMul;
- precision boundaries: explicit source Q/DQ;
- one logical FB with external-parameter splitting supported by the SDK.

GELab retains its required fused QKV, fixed 2-D visual RoPE, three DeepStack
outputs and main patch merger. The fixed profile is 512x288, or 576 visual
tokens before the 2x2 merger and 144 tokens afterward.

## FP16 baseline

The selected artifact is
`model/vision/single-g32x18-fp16/vision_g32x18-fp16.fb`:

- bytes: 827,717,284;
- SHA256: `4343170f7bfef7b88fd69fdb232676712426688e5a51c13ea17275a4185c3704`;
- one input and four outputs;
- no quantization tensors.

On four COCO holdouts against staged PyTorch FP32, main output cosine was
0.948346 minimum and 0.965567 mean. DeepStack minimum cosine was 0.994972,
0.988232 and 0.980380. A separate FP32 BCS semantic control on the first
holdout matched staged PyTorch FP32 at main cosine 0.999999999971 and maximum
absolute error 0.000110, proving that the BCS layout and patch-merger reorder
are correct. The remaining loss is the current software executor's FP16
accumulation path. It still misses the configured 0.99 gate and must be
rechecked on the target NPU.

The four-image metrics are bit-for-bit identical to the earlier 1xS Conv
reshape experiment. Balanced HxW therefore changes the exported topology and
NPU scheduling surface without changing the software-executor outputs.

The generated `topology.audit.json` covers all 96 block projections plus eight
merger projections: 104 FP16 Conv1x1 nodes, zero direct pre-Transpose nodes and
zero invalid entries. Pixel patches transpose once at graph entry, attention
transposes V/output for QK/AV semantics, and mergers reorder tokens only at
their shuffle/output boundaries; these are not per-projection conversions.

## Strict all-linear W8A8 with 64-token buckets

The experimental artifact is
`model/vision/single-g32x18-w8a8-linear-fp16-vit-tb64/vision_g32x18-int8-linear-fp16-fppatch-fpmerger-tb64.fb`:

- bytes: 530,724,868;
- SHA256: `30782bb6c99578b91490f46adf29339e595c87e088a69c48282a51b31c6f6f5a`;
- 96 quantized weights: 24 x QKV/O/FC1/FC2;
- all 24 MLP blocks quantized; no FP16 Top-K promotion;
- patch embedding, QK/Softmax/AV, residuals, LayerNorm and all mergers FP16;
- nine fixed token buckets of up to 64 tokens;
- 1,728 independently registered activation ranges.

For COCO holdout `000000109055.jpg`, fixed token buckets improved the strict
W8A8 result but did not pass:

| Output | BCS global-range W8A8 | BCS 64-token W8A8 | BCS FP16 |
| --- | ---: | ---: | ---: |
| Main | 0.495517 | 0.565580 | 0.960722 |
| DeepStack 0 | 0.971546 | 0.982900 | 0.995703 |
| DeepStack 1 | 0.641287 | 0.670866 | 0.990406 |
| DeepStack 2 | 0.667358 | 0.747271 | 0.984981 |

The BCS bucket graph contains 872 audited Conv1x1 nodes and zero direct
pre-Transpose nodes. It took 43.8 seconds to compile in the software executor
versus about 2.1 seconds for FP16, because four projections per block are
replicated over nine slices. It therefore remains an experiment and does not
replace the FP16 deployment profile.

## Diagnosis

Vit.py's fixed-role bucketing is effective for models with CLS/register versus
patch-token distributions. GELab has only spatial visual tokens. Its largest
layer-23 FC2 calibration value is 2457.18 and similar extremes occur across
most fixed positions, so position buckets cannot isolate the outliers. A
production A8 path needs content-dependent per-token online input
quantization, learned clipping/SmoothQuant, or selective floating-point
promotion; merely reducing the fixed bucket size is not sufficient evidence
of acceptable accuracy.

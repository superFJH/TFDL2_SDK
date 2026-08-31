# GELab-Zero-4B FP16 v1 accuracy report

Status: **bring-up only / vision accuracy gate failed**.

The selected visual artifact is now the Conv-native BCS graph at
`model/vision/single-g32x18-fp16/vision_g32x18-fp16.fb`. Four COCO validation
images outside the 64-image calibration split were processed by the official
processor at 512x288. References were generated layer-by-layer from the
checkpoint in PyTorch FP32 on CUDA; TFDL used the software executor.

| Output | Minimum cosine | Mean cosine | Maximum relative L2 |
|---|---:|---:|---:|
| Main | 0.948346 | 0.965567 | 0.321974 |
| DeepStack 0 | 0.994972 | 0.995668 | 0.106212 |
| DeepStack 1 | 0.988232 | 0.990900 | 0.153051 |
| DeepStack 2 | 0.980380 | 0.984241 | 0.197186 |

The same BCS source graph exported in FP32 was checked on the first holdout.
Its main output cosine against PyTorch FP32 was 0.999999999971 with maximum
absolute error 0.000110. This isolates the miss to the current TFDL software
FP16 numerical path rather than a BCS token/channel ordering error.

The graph topology audit covers all 96 block QKV/O/FC1/FC2 projections and
eight merger projections. All 104 are Conv1x1, and none has a Transpose as its
projection input. S=576 projections use 24x24 and merger S=144 uses 12x12;
1xS is reserved for prime sequence lengths. Pixel patches transpose once at
the graph ABI; attention and the four patch mergers retain only their
mathematically required reorderings.

The merged Mage-compatible W8A8 ONNX decoder previously passed its complete
runtime ABI smoke test: 75 inputs, 73 outputs, FP32 logits `[1,151936]`, FP16
present K/V, and all finite values. With a zero past of length four, ORT CPU
loaded in 5.50 seconds and executed one token in 0.102 seconds. A second
integration smoke used the real 169-token COCO prompt, 36 reference FP16 KV
prefixes and the actual seed token; it produced `The image` and ran the first
ORT continuation step in 0.100 seconds.

No FP16 overflow, Inf or NaN was observed. The vision miss means this profile
must not be represented as production-accuracy approved. Re-run the same
comparison on the actual NPU because hardware FP16 convolution/matmul
accumulation may differ from the software executor.

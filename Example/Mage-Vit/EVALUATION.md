# Mage-ViT conversion evaluation

Evaluation date: 2026-08-11

## Setup

- Checkpoint: `microsoft/Mage-VL`, joint `model.visual.*` weights, 24 layers.
- Input: Microsoft `examples/soccer-broadcast.mp4`.
- Codec path: official `codec-video-prep` 0.2.5 output, converted without
  resampling into the Mage-Vit PPM/position manifest.
- Fixed graph profile: one 288x512 canvas, 576 patches, 144x2560 output.
- Calibration: all 32 canvases from this video.
- Accuracy reference: the graph's PyTorch implementation loaded from the same
  checkpoint. TFDL tests used the CPU executor because the local NPU driver is
  not loaded.

The full floating TFDL graph reached cosine `0.99999999999056` versus PyTorch,
with maximum absolute error `3.6538e-5`. This verifies the checkpoint mapping,
patch ordering, 3-D RoPE and merger implementation before quantization.

## Speed-oriented INT8/FP16 sweep

The `int8-fp16-topk` profile keeps LayerNorm, residual streams and the merger
in FP16. Attention and MLP compute remain INT8 unless their residual-branch
output range is in the global Top-K. Selected Attention candidates restore
only the output projection; selected MLP candidates restore the complete MLP.

| Top-K | Selected FP16 branches | Bytes | MiB | Cosine |
|---:|---|---:|---:|---:|
| 0 | none | 362522020 | 345.7 | 0.737676 |
| 1 | MLP 14 | 370817840 | 353.6 | 0.795319 |
| 2 | MLP 14, MLP 23 | 379113660 | 361.6 | 0.810229 |
| 4 | MLP 14, 23, 15, 13 | 395705300 | 377.4 | 0.817678 |
| 8 | seven MLPs, Attention projection 23 | 421622848 | 402.1 | 0.821747 |

Top-K=2 is the smallest tested graph above the requested cosine 0.8 and is the
default. Its two selected ranges are `layers.14.fc2=462.672` and
`layers.23.fc2=405.499` by absolute maximum.

## Top-K=2 sampled-canvas check

| Canvas index | Cosine | Relative L2 |
|---:|---:|---:|
| 0 | 0.810229 | 0.737438 |
| 8 | 0.807160 | 0.820359 |
| 16 | 0.822964 | 0.713476 |
| 24 | 0.822919 | 0.713835 |

Cosine range is `0.807160-0.822964`; mean is `0.815818`. The output tensor is
FP16. Tensor audit reported 877 UINT8 tensors, 167 FP16 tensors and two FP32
RoPE inputs, plus one merged-tensor audit entry that the SDK could not query.

These measurements establish conversion fidelity, not final task accuracy or
NPU latency. The four evaluation canvases come from the calibration video, so
a production decision should repeat calibration/evaluation on disjoint videos
and measure end-to-end Qwen answers. Hardware execution is still pending:
opening `thinkforce0` currently fails with `please insmod driver`.

## H×S QK Requant update (2026-08-13)

The exporter now calibrates one QK range per `(head, query)` row. Its source
graph contains an identity-code Requant with score qscale exactly
`QK_qscale/8`; QuantizeLite encodes the graph without rewriting it. All 24
layers expose 9216 QK and score qinfo entries; Softmax output remains UINT8
with one ordinary qinfo and uses `AttnSoftmaxImpl` online quantization. The
largest exported MatMul
postscale multiplier was 0.038776, so no Mage QK row needed the default 0.99
range floor, but the strict-below-one guard remains part of the exporter.

The scalar and H×S calibrations used the same 32 soccer canvases. Their 441
shared scalar ranges were bit-identical; the new JSON only added 48 row-range
entries (QK and scaled score for 24 layers).

| Canvas | Scalar QK cosine | H×S QK cosine | Scalar rel-L2 | H×S rel-L2 |
|---:|---:|---:|---:|---:|
| 0 | 0.815780 | 0.822072 | 0.726568 | 0.670338 |
| 8 | 0.807231 | 0.821665 | 0.818144 | 0.746113 |
| 16 | 0.819840 | 0.841313 | 0.693446 | 0.635745 |
| 24 | 0.816387 | 0.828018 | 0.718323 | 0.704569 |
| Mean | 0.814810 | 0.828267 | 0.739120 | 0.689191 |

The H×S FB grew from 361.6 MiB to 368.3 MiB. CPU-executor time rose from about
30.8 s to 37.6 s per canvas; this is diagnostic execution and is not an NPU
latency measurement.

## Source Q/DQ migration (2026-08-13)

FP16 residuals/LayerNorms, the layer 14/23 Top-K MLP bypasses, and the complete
merger are now constructed directly in `build_tfdl_graph`. The two legacy
post-quant Modify implementations were removed. The only remaining Modify sets
Softmax `outputDataType=TFDtypeUint8` for `AttnSoftmaxImpl`.

The exported graph contains no `PostQuant_` or `PostDeQuant_` tensors. Its
dtype audit remains 877 UINT8, 167 FP16 and three FP32 tensors (raw-pixel
Placeholder plus two RoPE inputs). On canvases 0/8/16/24, cosine and relative
L2 matched the preceding post-quant H×S graph exactly, with zero delta in all
eight reported metrics.

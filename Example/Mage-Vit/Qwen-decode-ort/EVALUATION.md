# ONNX Runtime decoder evaluation

## Test platform

- CPU: 40-core Arm Cortex-A77, four DDR5 channels
- ISA: Armv8.2 FP16 and dot-product; no I8MM/SVE/SME
- Runtime: ONNX Runtime CPU 1.23.2, NumPy 2.2.6
- Text checkpoint: Mage-VL's Qwen3-4B-Instruct-2507, 36 layers
- Quantization: per-output-channel S8 weights and dynamic U8 activations
- KV: external FP16, 8 heads x 128 dimensions

## Build and graph audit

The complete build exports one FP32 layer at a time, quantizes it, deletes the
temporary graph and finally composes a single ORT model. Every decoder layer
contains exactly seven integer MatMuls: Q/K/V/O and gate/up/down. QK, Softmax
and AV stay floating point. The LM head is also W8A8.

The final graph has 5,344 nodes, 904 initializers, 75 inputs and 73 outputs.
Its operator audit contains 253 `MatMulInteger` nodes (`36*7 + LM head`), 72
floating `MatMul` nodes (QK/AV), 36 Softmax nodes and no Tile. The artifact is
about 4.04 GB including per-layer graph metadata. Session cold load is
approximately 5.6 seconds.

The generated artifact is `/tmp/mage-qwen3-ort-w8a8`. It is intentionally not
checked into the repository because its external weight files are several GB.

## Correctness

The first full-model check used the existing complete seq=4 NPU prefill bundle.
ORT and FP32 PyTorch imported exactly the same FP16 K/V and consumed the same
teacher-forced token on every step.

| Metric | Result |
|---|---:|
| Compared decode steps | 3 |
| Mean logits cosine | 0.995867 |
| Top-1 agreement | 3/3 |
| Mean Top-10 overlap | 9.33/10 |
| New K/V minimum layer cosine, step 0 | 0.980130 / 0.955344 |
| New K/V minimum layer cosine, step 1 | 0.997737 / 0.990466 |
| New K/V minimum layer cosine, step 2 | 0.996059 / 0.984043 |

This verifies the external-cache layout, GQA head mapping, RoPE position,
first-token handoff and cache append semantics. The generated text from this
particular bundle is not meaningful because its four-token prompt was a
synthetic calibration input.

Detailed results are in
`/tmp/mage-qwen3-ort-w8a8/ort_vs_torch_flat_seq4_fp32.json`.

## CPU performance

The final flattened-GQA graph produced:

| Threads | seq=4 tok/s | seq=869 tok/s |
|---:|---:|---:|
| 8 | 10.18 | 7.70 |
| 16 | 10.52 | 8.45 |
| 24 | 10.02 | 8.34 |
| 32 | 10.03 | 8.22 |
| 40 | 8.60 | 7.92 |

Sixteen threads are the best point on this CPU; using every core increases
synchronization overhead. With a synthetic seq=869 cache, physical KV-head
expansion produced only 4.66 tok/s; flattened GQA improves that by about 81%.

The quantized final RMSNorm/LM-head subgraph is also extracted as
`final_head.w8a8.onnx` and kept resident for the NPU-prefill first token. On
the deployment-job final hidden reproduced locally, the checkpoint-float path
took 0.621 s while the warm standalone W8A8 head took 7.4 ms. Logits cosine was
0.998094 and both paths selected token 87140. Target ARM timing remains to be
measured; the standalone graph reuses the existing 371 MiB external data shard.

Two no-expansion GQA layouts were then tested per layer at seq=869:

| Attention layout | layer latency, 16 threads |
|---|---:|
| Repeat 8 KV heads to 32 | 5.35 ms |
| 5-D broadcast MatMul | 9.26 ms |
| Flattened `(batch*KV_H)` 3-D MatMul | 2.63 ms |

The flattened layout is bit-exact to the repeat layout and is the final graph
structure.

The seq=869 cache used for this performance-only test contains zeros in layers
and must not be interpreted as a task-accuracy measurement.

## Real soccer-video prompt

The existing soccer shard contains 576 INT8/FP16 Mage-ViT visual tokens and an
869-token assembled prompt asking `Describe this video.` Only two NPU prefill
layers existed for that sequence length, so `export_reference_prefill.py` was
used to produce a complete PyTorch FP32 cache in the exact production FP16 ABI.
This isolates and validates the decoder; it is not a substitute for the final
36-layer NPU prefill run.

Over the first four teacher-forced decode steps:

| Metric | Result |
|---|---:|
| Mean logits cosine | 0.991667 |
| Top-1 agreement | 4/4 |
| Mean Top-10 overlap | 9.0/10 |
| ORT speed during comparison | 8.05 tok/s |

For an independent greedy 64-token run, ORT produced 8.18 tok/s versus 0.815
tok/s for the retained FP32 PyTorch cache-decode path, a 10.03x speedup. The
greedy streams shared the first six tokens and then diverged, while both gave a
correct semantic description of a sports commentator with a microphone in a
stadium. The ORT result began:

> The video clip shows a man speaking into a microphone labeled "BBC Sport."
> The background is a stadium filled with spectators.

Detailed reports:

- `/tmp/mage-qwen3-ort-w8a8/ort_vs_torch_soccer_seq869_fp32.json`
- `/tmp/mage-qwen3-ort-w8a8/decode_soccer_seq869_t16.json`
- `/tmp/mage-qwen3-ort-w8a8/decode_soccer_seq869_torch_fp32.json`

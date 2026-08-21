# Qwen3 W8A8 decode with ONNX Runtime

This directory supplies the CPU decode half of the hybrid pipeline:

```text
TFDL/NPU Qwen prefill
  -> FP16 K/V [1,H,S,D] + first-token logits
  -> one-time cache transpose to token-major [1,S,H,D]
  -> ONNX Runtime one-token decoder (W8A8 projections/MLP/LM head)
  -> append one FP16 K/V token per layer
```

The runtime never repeats prompt prefill. The first generated token comes from
`last_token_logits.npy`; ORT consumes that token at `position=S` to produce the
second generated token.

## Quantization profile

The first implementation uses ORT dynamic quantization:

- Q/K/V/O, gate/up/down and LM-head constant-weight MatMuls: per-output-channel
  S8 weights with dynamically generated U8 activations (W8A8);
- RMSNorm, RoPE, QK, Softmax, AV and residual streams: FP32;
- external KV input and one-token KV output: FP16.

Attention keeps the eight KV heads without materializing a 32-head cache.
`batch*KV_H` is flattened into the MatMul batch dimension and the four query
repeats become its M dimension. This produces the same result as expanding K/V,
but uses ORT's efficient 3-D batched MatMul instead of its slower 5-D broadcast
path. A fused GroupQueryAttention/custom CPU kernel remains a later option.

This is deliberately a correctness-first CPU decoder. Dynamic activation
quantization avoids a fixed decode calibration range and maps well to the
Cortex-A77 dot-product instructions. Static QDQ W8A8 and fused/paged attention
can be evaluated after this ABI is aligned.

## Environment

Use a NumPy-2-compatible CPU ONNX Runtime. The tested environment uses
`onnxruntime==1.23.2` on Linux AArch64:

```bash
source .venv-tfdl-linux/bin/activate
pip install --upgrade 'onnxruntime>=1.22,<2'
```

## Build the decoder

The filesystem does not need enough free space for a complete FP32 Qwen ONNX.
Every layer is exported and quantized independently; its temporary FP32 graph
is then removed. One top-level graph references the per-layer external INT8
weight files, so decode still creates only one ORT session.

```bash
python Example/Mage-Vit/Qwen-decode-ort/build_ort_qwen_decoder.py \
  --model-path /home/tf/.cache/modelscope/models/microsoft--Mage-VL/snapshots/master \
  --output-dir /tmp/mage-qwen3-ort-w8a8
```

The build is resumable. Existing quantized layer files are reused; pass
`--force` to rebuild them. The builder removes its exact external-data target
before a forced rewrite so ORT cannot append stale weights. For artifacts made
by an older revision, `--compact-only` rewrites external data and recomposes
the top-level graph without exporting/quantizing again. A real-weight layer
smoke test can be produced with:

```bash
python Example/Mage-Vit/Qwen-decode-ort/build_ort_qwen_decoder.py \
  --model-path /models/Mage-VL \
  --output-dir /tmp/mage-ort-layer0 \
  --layer 0 --skip-merge
```

Output ABI:

```text
hidden                         FP32 [1,1,2560]
position_sin/cos               FP32 [1,1,1,128]
past_key_values.N.key/value    FP16 [1,past,8,128]

logits                         FP32 [1,151936]
present.N.key/value            FP16 [1,1,8,128]
```

## Decode an existing NPU prefill bundle

```bash
python Example/Mage-Vit/Qwen-decode-ort/decode_ort.py \
  --model-path /models/Mage-VL \
  --decoder-dir /tmp/mage-qwen3-ort-w8a8 \
  --prompt-dir /tmp/mage-prompt \
  --prefill-dir /tmp/mage-prefill \
  --threads 16 \
  --max-new-tokens 128 \
  --output-json /tmp/mage-prefill/ort_decode.report.json
```

The report separates model/session load, one-time NPU cache conversion and ORT
step latency. `ort_tokens_per_second` counts only CPU decoder steps; the seed
token supplied by NPU logits is reported separately.

For numerical alignment, teacher-force the PyTorch reference and ORT with the
same tokens from the identical imported NPU KV:

```bash
python Example/Mage-Vit/Qwen-decode-ort/evaluate_ort_decoder.py \
  --model-path /models/Mage-VL \
  --decoder-dir /tmp/mage-qwen3-ort-w8a8 \
  --prompt-dir /tmp/mage-prompt \
  --prefill-dir /tmp/mage-prefill \
  --reference-dtype float32 \
  --steps 4 --threads 16
```

It reports logits cosine/Top-1/Top-10 and the mean/minimum new-K/V cosine over
all 36 layers at every decode step.

If a sequence bucket does not yet have a complete 36-layer NPU bundle, create
a correctness-only bundle with the identical ABI before testing the decoder:

```bash
python Example/Mage-Vit/Qwen-decode-ort/export_reference_prefill.py \
  --model-path /models/Mage-VL \
  --prompt-dir /tmp/mage-prompt \
  --output-dir /tmp/mage-reference-prefill \
  --dtype float32
```

This helper is strictly an A/B tool; production prefill remains TFDL/NPU.

## Complete video-to-text command

After the FFmpeg frontend, vision FB and fixed-sequence Qwen prefill FB stack
are built, all stages can be run with one command:

```bash
python Example/Mage-Vit/run_ort_pipeline.py \
  --model-path /models/Mage-VL \
  --video sample.mp4 \
  --frontend-bin /tmp/megavit-build/megavit_frontend \
  --vision-fb /tmp/mage_vit.quant.fb \
  --qwen-fb-dir /tmp/qwen-seq869-topk2 \
  --decoder-dir /tmp/mage-qwen3-ort-w8a8 \
  --output-dir /tmp/mage-ort-run \
  --question 'Describe this video.' \
  --hardware --threads 16
```

For the 869-token validation bucket used by this example, add
`--target-canvases 4`. When no NPU device is exposed, omit `--hardware` and
pass `--executor-config Example/Mage-Vit/runconfig.software.json`; this runs
the same visual and prefill FBs through the TFDL software executor.

Use `--skip-frontend` or `--skip-prefill` to resume from an existing stage
directory.

## Tests

The small-model test covers dynamic past length, W8A8 quantization, graph
composition, exact input/output names and FP16 present KV:

```bash
python Example/Mage-Vit/Qwen-decode-ort/test_ort_decoder.py
```

Measured graph, precision and Cortex-A77 results are recorded in
[`EVALUATION.md`](EVALUATION.md).

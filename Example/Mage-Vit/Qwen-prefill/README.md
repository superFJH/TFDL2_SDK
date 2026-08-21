# Qwen prefill on TFDL/NPU

This directory implements one fixed-sequence FB per Qwen3 decoder layer. The
default quantized attention path is:

```text
FP16 hidden/RMSNorm
  -> UINT8 Q/K/V projections
  -> FP16 QK RMSNorm + ApplyRope (K/V cache stays FP16)
  -> UINT8 grouped-GQA QK MatMul with H*S qinfo
  -> identity Requant with H*S qinfo scaled by 1/sqrt(128)
  -> ARM CausalMaskSoftmax (direct H*S qinfo, UINT8 [0,1])
  -> UINT8 grouped-GQA AV MatMul
  -> UINT8 projections / FP16 residual
```

The production precision plan follows `ConvertTools/python/example/Vit.py`:
source-level Q/DQ boundaries, FP16 Norm/residual/cache tensors, H*S QK plus an
identity scale-Requant, global Top-K floating branches, and Tok hybrid. For
Qwen, the stable default Tok hybrid splits every non-Top-K gated MLP into
contiguous prefix/visual-context/final-query branches. Every branch shares the
same per-output-channel UINT8 weights but receives independent scalar
activation qinfo. QKV remains shared by default because layer-0 A/B showed no
useful gain. Real-prompt audits found much larger K/V range separation in the
back half of the decoder, so `--token-hybrid-qkv-start-layer L` can apply the
same source-level split to Q/K/V from layer `L` onward. UINT8 AV/o_proj stays
shared because this SDK crashes on UINT8 Slice fan-out.

Unlike the standard-op ViT graph, the complete Qwen graph contains RMSNorm,
ApplyRope and ArmCausalMaskSoftmax CustomOps. The current SDK segfaults when
QuantizeLite scans that complete graph, including legacy scalar attention.
Projection weights are therefore encoded in isolated source `Quantize->Conv`
islands; the final Qwen graph still directly contains the complete explicit
Q/DQ topology and does not use `Modify`.

The SDK passes multi-channel activation metadata directly to Custom-op inputs.
The identity Requant therefore feeds the ARM op without a scalar transport
boundary or FP32 scale sidecar. The audit requires QK and scale-Requant raw
codes to be bit-exact and verifies all H*S score qinfo at the Custom boundary.

GQA is evaluated as `[KV_H, repeats*S, D] @ [KV_H, D, S]`, rather than
physically repeating K/V. Besides saving memory, this avoids an SDK crash in
the UINT8 `Slice -> repeated Concat` path.

## Build the addon

```bash
cmake -S AddonOps -B AddonOps/build
cmake --build AddonOps/build -j2
```

## Prepare a real video prompt

The visual frontend bundle contains `visual_embeddings.f32` and a manifest.
Assemble these with the exact tokenizer template and text embedding table:

```bash
python Example/Mage-Vit/Qwen-prefill/prepare_qwen_prefill_prompt.py \
  --model-path /models/Mage-VL \
  --bundle /tmp/mage-run \
  --question "Describe this video." \
  --pad-to-seq-len 1024 \
  --output-dir /tmp/mage-prompt
```

This writes `hidden.npy`, `position_ids.npy`, token IDs, mask, and metadata.
With `--pad-to-seq-len`, metadata records both `model_seq_len` and
`valid_seq_len`; padding is always a right suffix.

## Calibrate and build a 36-layer stack

Multiple `--prompt-dir` arguments are aggregated. Every QK range contains one
entry per `(head, query)` row and is unioned across prompts. When prompt
metadata has `valid_seq_len`, padding activations and future causal QK cells
are excluded. At least one calibration prompt must reach the final bucket row
or range export fails rather than silently using padding statistics.

```bash
python Example/Mage-Vit/Qwen-prefill/collect_qwen_prefill_ranges.py \
  --model-path /models/Mage-VL \
  --prompt-dir /tmp/mage-prompt-a \
  --prompt-dir /tmp/mage-prompt-b \
  --dump-ranges /tmp/qwen-seq869.ranges.json \
  --output-json /tmp/qwen-seq869.calibration.json

python Example/Mage-Vit/Qwen-prefill/build_qwen_prefill_stack.py \
  --model-path /models/Mage-VL \
  --seq-len 869 \
  --range-json /tmp/qwen-seq869.ranges.json \
  --calibration-report /tmp/qwen-seq869.calibration.json \
  --calibration-language en \
  --output-dir /tmp/qwen-seq869-topk2 \
  --outlier-top-k 2 \
  --attention-mode arm-causal-hxs \
  --prompt-dir /tmp/mage-prompt-a \
  --per-channel-qk-max-requant-multiplier 0.99 \
  --softmax-threads 0
```

`--outlier-top-k` is a global ranking over all 36 layers' `o_proj/down_proj`
absolute output ranges. Selected branches are FP16 in the source graph. The
default path does not call `Modify`.

Every exported layer is reopened with a non-frugal software executor and all
tensors returned by `TFContext.GetAllTensorNames()` are checked. Every UINT8
tensor must have finite, positive and consistently sized qmin/qmax/qscale/
zero-point arrays. The complete inventory is written to
`layer_XX.tensor-audit.json`; a missing qinfo aborts the build before the stack
manifest is completed. `--skip-tensor-audit` is only intended for SDK failure
diagnosis, not deployment exports.

`--prompt-dir` infers the Tok-hybrid boundaries from the first and last image
token. They can instead be supplied explicitly with
`--token-group-boundaries START END`. Tok hybrid requires a range JSON made by
the current collector because it consumes the saved `.tokens` per-token range
arrays.

The production baseline leaves `--token-hybrid-qkv-start-layer` unset. The
soccer S=898 accuracy profile uses `--token-hybrid-qkv-start-layer 12` as an
optional higher-precision preset. This improves later K/V caches and final
logits, but it is still a static calibration scheme: use a representative
multi-prompt range set and revalidate the start layer for another sequence
bucket. `--outlier-top-k 4` is the corresponding accuracy-oriented Top-K
preset; Top-K=2 remains the lower-FP16 baseline.

Static Tok-hybrid boundaries must also match the runtime image-token span.
Right-padding makes the tensor shape flexible, but does not move the
Slice/Concat boundaries baked into an FB.  Check a prepared prompt without
running the stack:

```bash
python Example/Mage-Vit/Qwen-prefill/run_qwen_prefill_stack.py \
  --model-path /models/Mage-VL \
  --fb-dir /models/qwen-prefill-s1024 \
  --prompt-dir /tmp/mage-prompt \
  --output-dir /tmp/unused \
  --check-prompt-only --require-token-group-match
```

The runner also prints `last_hidden_cos` during reference comparisons.  This
is the answer-bearing final query token; a flattened hidden cosine can remain
high, or collapse on visual outliers, without describing that token.

This distinction also explains the gap to the ORT W8A8 decoder. ORT's dynamic
quantized MatMul computes an activation scale for the current decode token at
runtime and keeps RMSNorm/RoPE/attention/residual arithmetic floating. The NPU
prefill graph uses fixed calibration qinfo over hundreds of heterogeneous
prefix, visual and query tokens. Both paths use W8 weights, but their
activation granularity and attention precision are not equivalent.

## Run the stack and export the decode ABI

```bash
python Example/Mage-Vit/Qwen-prefill/run_qwen_prefill_stack.py \
  --model-path /models/Mage-VL \
  --fb-dir /tmp/qwen-seq869-topk2 \
  --prompt-dir /tmp/mage-prompt-a \
  --output-dir /tmp/qwen-prefill-output \
  --compare-reference
```

Each layer's FP16 K/V is saved independently, followed by the final FP16
hidden state and last-token logits. The final RMSNorm/LM head is computed on
CPU or GPU with `--final-device`; the LM-head matrix is streamed in blocks.
The JSON report contains per-layer hidden/K/V drift, logits cosine, Top-1 and
Top-10 overlap. The K/V NPY files and manifest are the handoff for a
pure-decode runtime. The implemented CPU path is the W8A8 ONNX Runtime engine
in `../Qwen-decode-ort`; it imports these files without repeating prefill.

For the PyTorch reference decode handoff (without repeating prefill):

```bash
python Example/Mage-Vit/Qwen-prefill/decode_from_prefill_cache.py \
  --model-path /models/Mage-VL \
  --prompt-dir /tmp/mage-prompt-a \
  --prefill-dir /tmp/qwen-prefill-output \
  --device cpu --dtype bfloat16
```

This helper has the same `transformers>=4.57` requirement as `qwen3_bridge.py`.

For the W8A8 ONNX Runtime decoder:

```bash
python Example/Mage-Vit/Qwen-decode-ort/decode_ort.py \
  --model-path /models/Mage-VL \
  --decoder-dir /tmp/mage-qwen3-ort-w8a8 \
  --prompt-dir /tmp/mage-prompt-a \
  --prefill-dir /tmp/qwen-prefill-output \
  --threads 16
```

## Diagnostics and tests

```bash
# Scalar/H*S correctness and Cortex-A77 thread sweep
python Example/Mage-Vit/Qwen-prefill/test_arm_attention_ops.py \
  --sequence 128 --sequence 512 \
  --threads 1 --threads 4 --threads 8 --threads 16 --threads 40

# FP32/FP16 hidden- and Q-head-shape RMSNorm correctness/thread sweep
python Example/Mage-Vit/Qwen-prefill/test_arm_rmsnorm.py \
  --threads 1 --threads 4 --threads 8 --threads 16 --threads 40

# Inspect layer boundary codes, qinfo, probability and AV
python Example/Mage-Vit/Qwen-prefill/audit_qwen_prefill_attention.py \
  --model-path /models/Mage-VL --layer-id 0 \
  --fb /tmp/qwen/layer_00_seq_869.fb \
  --symbol-map /tmp/qwen/layer_00.symbols.json \
  --prompt-dir /tmp/mage-prompt-a \
  --output-json /tmp/layer0.audit.json

# Dependency-light contract checks
python Example/Mage-Vit/Qwen-prefill/test_qwen_prefill.py
```

Legacy float, QuantizeLite FP16-attention and SDK-PTQ builders are retained in
`build_qwen_prefill.py` for A/B diagnosis. The production default is
`arm-causal-hxs`.

Measured operator, layer, real-prompt and Top-K sweep results are recorded in
[`EVALUATION.md`](EVALUATION.md).

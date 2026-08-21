# Mage Qwen3 layer-0 PTQ reproducer

This directory contains the real first Qwen3 decoder layer from the local
Mage-VL checkpoint, exported as a cache-free, fixed-sequence TFDL graph.  It is
the floating-point input graph on which the current INT8 + FP16 boundary
rewrite fails.

## Files

- `mage_qwen3_layer0_seq4_fp32.fb`: self-contained FP32 TFDL graph and weights.
- `mage_qwen3_layer0_seq4.symbols.json`: logical-name to TFDL-symbol mapping.
- `mage_qwen3_layer0_seq4.ranges.json`: seed-1234 min/max activation ranges.
- `mage_qwen3_layer0_seq4.fp32_eval.json`: PyTorch/TFDL alignment report.
- `mage_qwen3_layer0_seq4.int8_fp16_modify.json`: the exact 24-entry mixed
  precision rewrite submitted after SDK PTQ.  It is written before `Modify`,
  so it remains available when the native call fails.

The FB SHA-256 is:

```text
680db3e9898faef70f6a42c5b687f5e711387ef22d72875e1ec543fc2ae75539
```

## Graph ABI

Model dimensions: hidden size 2560, query width 4096, 32 query heads, 8 KV
heads, head dimension 128, and MLP intermediate size 9728.

Inputs, in executor order:

1. `TFDL_Placeholder_0`: hidden state, `[1, 4, 2560]`, FP32.
2. `TFDL_Placeholder_1`: RoPE sine, `[1, 1, 4, 128]`, FP32.
3. `TFDL_Placeholder_2`: RoPE cosine, `[1, 1, 4, 128]`, FP32.
4. `TFDL_Placeholder_3`: causal additive mask, `[1, 4, 4]`, FP32.

Outputs:

1. `TFDL_ADD_2`: next hidden state, `[1, 4, 2560]`.
2. `TFDL_MUL_6`: layer K cache, `[1, 8, 4, 128]`.
3. `TFDL_MUL_7`: layer V cache, `[1, 8, 4, 128]`.

The FP32 graph is aligned against the exact PyTorch layer, including Qwen3
Q/K RMSNorm and GQA.  Hidden, K, and V cosine similarity are all effectively
1.0; full errors are recorded in the evaluation JSON.

## Reproduce the current failure

From the SDK root:

```bash
source .venv-tfdl-linux/bin/activate
export LD_LIBRARY_PATH=/home/tf/TFDL2_SDK/lib
python -X faulthandler Example/Mage-Vit/Qwen-prefill/build_qwen_prefill.py \
  --model-path /home/tf/.cache/modelscope/models/microsoft--Mage-VL/snapshots/master \
  --layer-id 0 \
  --seq-len 4 \
  --range-json Example/Mage-Vit/Qwen-prefill/repro/mage_qwen3_layer0_seq4.ranges.json \
  --dump-quant-fb /tmp/mage_qwen3_layer0_seq4_sdk_ptq.fb \
  --quant-mode sdk-ptq \
  --outlier-top-k 0 \
  --dump-modify-json Example/Mage-Vit/Qwen-prefill/repro/mage_qwen3_layer0_seq4.int8_fp16_modify.json
```

In the current SDK, `TFCalibration.Quantize` finishes, then `TFContext.Modify`
prints `Warning:Can't find Op NoOp` and throws `RuntimeError: Caught an unknown
exception!`.  No INT8 FB is emitted.  The likely compatibility issue is that
PTQ optimization folds at least one symbol addressed by the mixed-precision
rewrite into a `NoOp`; the exact offending entry still needs to be isolated.

## Intended INT8/FP16 partition

- SDK PTQ uses per-channel INT8 weights and the supplied activation min/max.
- Q/K/V, attention output, and ordinary MLP projections remain INT8.
- Q/K normalization, RoPE, GQA, and the QK matmul stay on the INT8 path.
- Attention scores are dequantized before the causal-mask add and Softmax;
  mask add and Softmax run in FP16.
- Probabilities are requantized for the AV matmul and output projection.
- Residual additions, input/post-attention RMSNorm, final hidden output, and
  exported K/V use FP16.
- With `top-k > 0`, attention-output and MLP-down branches are ranked by their
  absolute output range; only the largest global outliers are restored to
  FP16, following the policy used in `ConvertTools/python/example/Vit.py`.

The included ranges use a deterministic synthetic hidden state and are meant
only to reproduce the converter issue.  Production artifacts should collect
ranges from representative video/prompt prefill inputs for every sequence
bucket.

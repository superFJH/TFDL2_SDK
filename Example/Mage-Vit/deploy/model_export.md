# Exporting deployment models

[`../export_models.py`](../export_models.py) is the supported entry point for
rebuilding the artifacts in this directory from another Mage-VL checkpoint.
It orchestrates the existing vision, Qwen prefill and ONNX Runtime decoder
builders; the quantization implementation remains in those builders.

The exporter builds into `deploy/var/model-export/staging-models`, validates
the complete staged result, and publishes only successful components. With the
default output directory it also regenerates `deploy/assets.json`.

## Supported checkpoint ABI

"Your own model" currently means a fine-tuned or otherwise weight-modified
checkpoint with the Microsoft Mage-VL architecture and tokenizer ABI:

- Mage-ViT: 24 layers, hidden 1024, patch 16, spatial merge 2, output 2560.
- Qwen3: 36 layers, hidden 2560, intermediate 9728, 32 Q heads, 8 KV heads,
  head size 128.
- The checkpoint must include its tokenizer files and either
  `model.safetensors.index.json` or `model.safetensors`.

The script rejects a different architecture before loading weights. Arbitrary
ViT/Qwen shapes require graph and deployment-runtime changes; renaming such a
checkpoint is not sufficient.

At runtime, `MAGE_VL_MODEL_PATH` must point to the same checkpoint used here.
Prompt assembly still reads its tokenizer and embedding table. Mixing exported
FB/ONNX files with another checkpoint can produce fluent-looking but incorrect
or garbled output.

## Prerequisites

Run from the SDK root in the same Python environment used by the conversion
tools:

```bash
source .venv-tfdl-linux/bin/activate
python -m pip install ./Python
python -m pip install -r Example/Mage-Vit/deploy/requirements.txt
Example/Mage-Vit/deploy/build_target.sh
```

The default addon is `deploy/runtime/libTFDLAddOn.so`. The export host needs
enough memory and temporary disk for one decoder layer plus the final package;
`deploy/models` itself is roughly 12 GB for the current checkpoint.

The examples below use:

```bash
MAGE_MODEL=/models/microsoft--Mage-VL/snapshots/master
MAGE_EXPORT=Example/Mage-Vit/deploy/export_models.py
MAGE_CALIB=/data/mage-vl-calibration
mkdir -p "$MAGE_CALIB"
```

## 1. Create representative Canvas bundles

Generate frontend bundles from several videos that represent the expected
durations, motion, scenes and aspect ratios. Keep `target-canvases=4`, because
this deployment profile and its visual-token budget are built for four
canvases.

```bash
Example/Mage-Vit/build/megavit_frontend \
  --video /data/videos/soccer.mp4 \
  --target-canvases 4 \
  --output-dir "$MAGE_CALIB/frontend-soccer"

Example/Mage-Vit/build/megavit_frontend \
  --video /data/videos/indoor.mp4 \
  --target-canvases 4 \
  --output-dir "$MAGE_CALIB/frontend-indoor"
```

Each bundle must contain `manifest.json` and the PPM canvas files it names.
The exporter accepts repeated `--vision-calibration-bundle` arguments and
aggregates their ranges. `--max-vision-calib N`, when nonzero, limits the
number of canvases per bundle rather than globally.

## 2. Export the visual encoder

This phase is separate because prefill calibration should consume embeddings
produced by the newly quantized visual FB, not embeddings from an old model.

```bash
python "$MAGE_EXPORT" build \
  --model-path "$MAGE_MODEL" \
  --component vision \
  --vision-calibration-bundle "$MAGE_CALIB/frontend-soccer" \
  --vision-calibration-bundle "$MAGE_CALIB/frontend-indoor" \
  --device cuda \
  --no-update-assets \
  --force
```

The resulting graph is
`deploy/models/vision/mage_vit_288x512.int8_fp16_topk2.fb`. It uses the same
source-graph FP16 residual/merger, Top-K=2 bypass, H×S QK and identity
attention Requant policy as `python/build_mage_vit.py`. The exporter also runs
`GetAllTensorNames` qinfo validation and writes `vision/tensor-audit.json`.

Do not start the service between this phase and the final export: the new
vision FB and the old prefill/decoder may belong to different checkpoints.

## 3. Produce visual embeddings and prompt calibration samples

Run the new visual FB on each representative frontend bundle. Omit
`--hardware` for the TFDL software executor, or add it on an NPU export host.

```bash
python Example/Mage-Vit/python/export_mage_embeddings.py export \
  --model-path "$MAGE_MODEL" \
  --bundle "$MAGE_CALIB/frontend-soccer" \
  --backend tfdl \
  --fb Example/Mage-Vit/deploy/models/vision/mage_vit_288x512.int8_fp16_topk2.fb \
  --addon-path Example/Mage-Vit/deploy/runtime/libTFDLAddOn.so \
  --output-bundle "$MAGE_CALIB/embedding-soccer"
```

Prepare multiple natural questions and languages. Every prompt is right-padded
to the fixed S=1024 graph, while `metadata.json` records the real length.

```bash
python Example/Mage-Vit/Qwen-prefill/prepare_qwen_prefill_prompt.py \
  --model-path "$MAGE_MODEL" \
  --bundle "$MAGE_CALIB/embedding-soccer" \
  --question "Describe this video." \
  --system "You are a helpful assistant." \
  --pad-to-seq-len 1024 \
  --output-dir "$MAGE_CALIB/prompt-soccer-short"
```

H×S QK needs at least one calibration sample whose `valid_seq_len` is exactly
1024; ordinary right-padding does not calibrate those tail rows. The helper
below finds a tokenizer-valid filler length and prepares one non-padding tail
coverage sample from a real visual embedding bundle:

```bash
python "$MAGE_EXPORT" make-full-prompt \
  --model-path "$MAGE_MODEL" \
  --bundle "$MAGE_CALIB/embedding-soccer" \
  --system "You are a helpful assistant." \
  --output-dir "$MAGE_CALIB/prompt-soccer-full"
```

This filler sample is the minimum needed to cover all 1024 attention rows; it
does not replace natural short, medium and long questions in the calibration
set. Use several videos and both English/Chinese prompts if those are expected
in production.

The current Tok-hybrid graph has the stable prefix/rest boundary `[21]`. The
exporter checks that the first image token is offset 21 in every prompt. If the
system prompt or chat template changes that offset, regenerate every prompt,
pass the new fixed value with `--token-group-boundaries`, and update the
service's `system_prompt` consistently. Changing user question text alone does
not change this prefix boundary.

## 4. Export prefill and decoder

Pass every prepared prompt directory. The default produces the deployed
Top-K=4 profile and the Top-K=0 speed A/B profile from one shared range file.

```bash
python "$MAGE_EXPORT" build \
  --model-path "$MAGE_MODEL" \
  --component prefill \
  --component decoder \
  --prefill-prompt-dir "$MAGE_CALIB/prompt-soccer-short" \
  --prefill-prompt-dir "$MAGE_CALIB/prompt-soccer-full" \
  --calibration-language en \
  --device cuda \
  --force
```

Add more `--prefill-prompt-dir` and `--calibration-language` options for the
real dataset. Use `--device cpu` on a host without CUDA; range collection will
be substantially slower. The production defaults retained by the wrapper are:

```text
sequence length                 1024
attention                       ARM causal H×S
QK/Requant max multiplier       0.99
MLP token-group boundary        [21]
QKV token hybrid start layer    12
prefill profiles                Top-K=4 and Top-K=0
decoder                         ORT dynamic W8A8, FP16 external KV
```

To export only one prefill profile, pass `--prefill-top-k 4` or
`--prefill-top-k 0`. The stock deployment package expects both, so regenerate
the missing profile before refreshing `assets.json`.

If all prompt directories already correspond to the target visual FB, all
three components can be built in one invocation by omitting `--component` and
supplying both the vision bundles and prompt directories.

## 5. Validate and deploy

```bash
python Example/Mage-Vit/deploy/verify_assets.py

MAGE_VL_MODEL_PATH="$MAGE_MODEL" \
  python Example/Mage-Vit/deploy/app.py --validate-only
```

`verify_assets.py` checks the package hashes, 36 FBs in each prefill profile
and 37 decoder external-data shards. Each prefill builder additionally refuses
to publish if any UINT8 tensor lacks complete qinfo. The full conversion log,
ranges, calibration report and exact child commands remain under
`deploy/var/model-export`; `deploy/models/export.report.json` records the
checkpoint and calibration provenance.

After manually copying or replacing artifacts, validate and rewrite hashes
with:

```bash
python "$MAGE_EXPORT" refresh-assets
python Example/Mage-Vit/deploy/verify_assets.py
```

`refresh-assets` is intentionally strict: both Top-K profiles, the vision FB,
the decoder, final head, all 72 prefill audits and external-data shards must
already be complete. Newly exported vision directories also contain their own
qinfo audit; the refresh command remains compatible with older valid packages
that predate that one file.

## Safety, dry runs and recovery

Existing deployed directories are never replaced without `--force`. Even
with `--force`, conversion first finishes in the staging directory and only
then swaps the selected component directories. A failed child process leaves
the currently deployed model untouched and points to the complete log.

Inspect the exact commands and calibration checks without converting weights:

```bash
python "$MAGE_EXPORT" build \
  --model-path "$MAGE_MODEL" \
  --vision-calibration-bundle "$MAGE_CALIB/frontend-soccer" \
  --prefill-prompt-dir "$MAGE_CALIB/prompt-soccer-short" \
  --prefill-prompt-dir "$MAGE_CALIB/prompt-soccer-full" \
  --device cpu \
  --dry-run
```

If publication is interrupted, inspect any exact sibling named
`.export-incoming` or `.export-backup` before retrying. The script will not
silently delete such recovery directories.

# Mage-Vit NPU deployment

This directory is the movable deployment package for the validated fixed
bucket:

```text
video
  -> C++ FFmpeg codec frontend
  -> Mage-ViT H*S INT8/FP16 Top-K=2 FB on NPU
  -> Qwen prefill fixed S=1024, QKV Tok hybrid from layer 12, Top-K=4 on NPU
  -> FP16 KV cache
  -> Qwen3 W8A8 ONNX Runtime decode on CPU
```

The package contains all deployed FB and ONNX artifacts. The original
Microsoft Mage-VL checkpoint is not duplicated; set `MAGE_VL_MODEL_PATH` to
its snapshot directory. The active field-test bucket is calibrated for four
canvases and accepts a custom user question of at most 120 tokenizer tokens.
The complete codec-selection, Canvas-packing and visual-token algorithm is
documented in
[`../VIDEO_FRONTEND_CANVAS_AND_TOKENS.md`](../VIDEO_FRONTEND_CANVAS_AND_TOKENS.md).
Its Tok-hybrid graph has one stable boundary at token 21: the fixed system/chat
prefix is one group and the complete visual/question/padding remainder is the
other. Prompts are right-padded to S=1024 inside the NPU graph; only the real
prefix is exported to the CPU decoder. The current ranges contain one soccer
prompt and one real prompt reaching S=1024, so this is a generalization test
profile rather than a production calibration set. A request whose assembled
prompt exceeds S=1024 is rejected before prefill.

## Contents

```text
deploy/models/vision/                         one Mage-ViT FB
deploy/models/qwen-prefill/s1024-flex-topk4-qkv12/  default 36-layer `[21]` Top-K=4 profile
deploy/models/qwen-prefill/s1024-flex-topk0-qkv12/  speed A/B 36-layer `[21]` Top-K=0 profile
deploy/models/qwen-decode-ort/w8a8/            ORT decoder + standalone W8A8 final head
deploy/runtime/libTFDLAddOn.so                 RoPE/MaskSoftmax addon
deploy/deployment.json                         fixed-bucket service config
deploy/app.py                                  Flask UI + API
deploy/persistent_runtime.py                   resident NPU/ORT runtime
deploy/templates/ and deploy/static/           browser demo
```

The package contains the S=1024 Top-K=4 deployment profile and its Top-K=0
speed/accuracy A/B. Run the integrity check after copying it to another
machine:

```bash
python3 Example/Mage-Vit/deploy/verify_assets.py
```

To rebuild these FB/ONNX artifacts from another compatible Mage-VL
checkpoint, use `deploy/export_models.py`. The complete calibration, staged
publication and validation procedure is in
[`models/README.md`](models/README.md).

Both profiles use the channel-major QKV Tok-hybrid layout from layer 12. The
layer 12 graph has 11 Transpose and 20 Reshape nodes (previously 18 and 23),
while preserving bit-exact hidden/K/V output in the software same-input test.

The default `deployment.json` selects Top-K=4. To run the otherwise identical
Top-K=0 speed/accuracy profile, select its deployment config before startup:

```bash
MEGAVIT_DEPLOY_CONFIG=deploy/deployment.topk0.json \
  Example2/Mage-Vit/deploy/start.sh
```

`/v1/models` then reports profile `s1024-flex-topk0-qkv12` and
`outlier_top_k: 0`. Stop the service before switching profiles because all 36
contexts are constructed once at process startup.

The resident runtime also loads `final_head.w8a8.onnx` for the first token.
This replaces the checkpoint-float CPU RMSNorm/LM-head path without changing
the later decoder. Four vision Executors share one TFContext and process the
four canvas images concurrently; set `vision_workers` to `1` for a serial A/B.
It can also be overridden without editing JSON via
`MEGAVIT_VISION_WORKERS=1`.

All Mage-Vit NPU entry points use the shared Executor configuration in
`python/npu_executor_config.py`. Vision uses `Core=[-1]` and `useCache=true`;
the 36 resident prefill layers use `Core=[0,1,2,3]` and `useCache=false`.
Both use `FrugalMode=true`, `cpuLimit=16`, `MakeAlign=true` and
`AttnSoftmaxImpl=true`. The effective dictionaries are exposed under
`persistent_runtime.startup` in `/health` so target-machine configuration can
be checked without reading the source.

## Target-machine setup

The whole SDK tree should be copied because the frontend and Python wrapper
link against the SDK runtime. On Debian/Ubuntu ARM64:

```bash
sudo apt-get install -y \
  build-essential cmake pkg-config python3-dev python3-venv \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev

python3 -m venv .venv-tfdl-linux
source .venv-tfdl-linux/bin/activate
python -m pip install --upgrade pip pybind11
python -m pip install ./Python
python -m pip install -r Example/Mage-Vit/deploy/requirements.txt

Example/Mage-Vit/deploy/build_target.sh
```

`build_target.sh` fails if CMake silently selected either the FFmpeg or TFDL
stub. It also rebuilds the native TFDL2 Python wrapper with the same Python
used for deployment and verifies that `deploy/runtime/libTFDLAddOn.so` can be
registered. When the service uses a non-default interpreter, select it
explicitly:

```bash
MEGAVIT_PYTHON=/root/thinkforce/bin/python3 \
  Example/Mage-Vit/deploy/build_target.sh
```

A valid upstream-FFmpeg target prints:

```json
{"ffmpeg":true,"tfdl":true,"patched_bitcost":false}
```

`patched_bitcost=false` is runnable and uses the documented
motion-vector/pixel-residual fallback. Link the shim implementing
`mage_ffmpeg_get_bitcost` to obtain the intended codec-bitcost frontend; the
health response reports this distinction as a warning.

If an older deployment fails with `Can't find Op ArmCausalMaskSoftmax`, make
sure `run_ort_pipeline.py` passes the deployment addon to
`run_qwen_prefill_stack.py` as `--addon-path`. Loading the addon in the C++
vision process is not enough: custom-op registration is process-local. The
absolute addon path serialized in an FB may refer to the build machine; it is
only a fallback and must not be used as the deployment path. After syncing the
source, rerun `build_target.sh` and restart the service.

## Start the service

```bash
export MAGE_VL_MODEL_PATH=/models/microsoft--Mage-VL/snapshots/master
export MEGAVIT_BIND=0.0.0.0:5000
Example/Mage-Vit/deploy/start.sh
```

`MAGE_VL_MODEL_PATH` overrides `model_path` in `deployment.json`. If the JSON
contains a valid absolute model path, the environment variable can be omitted.

The launcher performs a lightweight deployment validation first, then starts
one Gunicorn worker.  During worker startup it registers the addon and creates
one resident Vision Context/Executor, all 36 Qwen prefill Context/Executors,
the tokenizer/embedding readers and one ORT decoder session.  These objects
remain alive and are reused by every request.  A request starts only the C++
FFmpeg/canvas frontend without `--model`; vision inference, prompt assembly,
prefill and decode all run in the Flask worker.

Do not increase the worker count: every process would load another complete
NPU/ORT runtime and could submit concurrently to the same NPU. HTTP threads
are safe; the service admits one inference at a time and returns HTTP 429
while busy.  Do not add Gunicorn `--preload`: NPU objects must be constructed
inside the serving worker and must never be inherited across `fork()`.

Check readiness:

```bash
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/v1/models
```

Open `http://TARGET-IP:5000/` for the interactive demo. It supports drag/drop
video upload, custom questions, live stage status and token-by-token output.
The timing cards report:

- **TTFT**: request inference start through FFmpeg, resident vision NPU,
  prompt assembly and resident NPU prefill until the first visible token.
  Upload transfer time and one-time service startup are excluded.
- **Decode**: first visible token until the complete answer. This includes ORT
  decode only; the ORT session is already initialized at service startup.
- **Throughput**: pure measured ORT decode steps per second.
- **Total**: complete server-side inference pipeline time.

Run a video request with any question that fits the bucket:

```bash
curl -X POST http://127.0.0.1:5000/v1/video/generate \
  -F video=@soccer-broadcast.mp4 \
  -F 'question=What color is the microphone?' \
  -F max_new_tokens=128
```

The response contains generated text/tokens, stage timings and ORT decode
throughput. Uploaded video, visual tensors and KV cache are removed after a
successful or failed request unless `keep_jobs` is set to `true` in
`deployment.json`.

In persistent mode the NPU KV bundle is handed directly to the resident ORT
decoder in memory.  The 36 K/V `.npy` pairs, prompt arrays and JSON reports are
written only when `keep_jobs=true`; normal requests avoid that temporary disk
I/O entirely.

Set `"persistent_runtime": false` only when debugging the older subprocess
pipeline.  The persistent mode is the deployment default.  `/health` reports
the startup duration, resident Context/Executor counts and ORT session load
time under `persistent_runtime.startup`.

For programmatic streaming, POST the same multipart fields to
`/v1/video/generate/stream`. The response is newline-delimited JSON with
`stage_start`, `stage_done`, `token`, `done` or `error` events. The blocking
`/v1/video/generate` endpoint remains available. When placing the service
behind Nginx, keep response buffering disabled; the Flask response already
sets `X-Accel-Buffering: no`.

## Failure logs

Every request writes merged stdout/stderr to
`deploy/var/jobs/<job-id>/pipeline.log`. The web UI displays the last 12 KiB
when a streamed request fails. To retain the complete input, intermediate
tensors and log after completion or failure, set `"keep_jobs": true` in
`deploy/deployment.json` and restart the service. Inspect retained logs with:

```bash
find Example/Mage-Vit/deploy/var/jobs -name pipeline.log \
  -printf '%T@ %p\n' | sort -nr | head
tail -n 500 Example/Mage-Vit/deploy/var/jobs/<job-id>/pipeline.log
```

Return `keep_jobs` to `false` for normal deployment because retained videos,
embeddings and KV caches consume significant disk space.

## Flexible prompt inside a fixed graph bucket

The H*S attention rows and source graph shapes remain sequence-specific, but
the tokenizer output may be shorter than S=1024. `prepare_qwen_prefill_prompt`
right-pads IDs, hidden states and RoPE inputs. Causal attention guarantees
that real query rows cannot see the right padding. After all 36 layers,
`run_qwen_prefill_stack` uses `hidden[valid_seq_len-1]` for the first-token LM
head and writes only `K/V[:valid_seq_len]`; ORT therefore resumes at the real
position rather than position 1024.

The API performs a cheap question-token check before codec/NPU work and then
does an exact full-template capacity check during prompt assembly. The 120
token question limit leaves margin for timestamp tokenization. An oversized
question returns HTTP 422.

Calibration used English short/detail prompts, a Chinese prompt, and a real
1024-token prompt so every H×S tail row was observed. Padding activations and
causally invisible future QK cells are excluded from ranges. A different
canvas count still needs another 36-layer bucket because it changes the
visual-token layout. Multiple sequence buckets can be routed later to reduce
the S=1024 compute overhead for short questions.

Software-executor validation on the soccer video produced logits cosine
0.93666 with matching Top-1 for `Describe this video.`. A Chinese 918-token
prompt produced cosine 0.86777 and Top-10 overlap 8/10; its first two logits
swapped, but generation remained coherent. The fixed-question S=898 profile
was more accurate (0.98402), so applications requiring near-reference
ranking should add narrower buckets rather than widening this one further.

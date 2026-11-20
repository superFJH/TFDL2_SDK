# Breeze-TTS-2.cpp source provenance

Source: https://github.com/HoppouAI/Breeze-TTS-2.cpp

Pinned revision: `155e60818433ea6322367aff8f8436ba790f6fbe` (2026-11-16).
The accompanying `LICENSE` is the upstream Apache-2.0 source license. No model
weights are vendored. Weight licensing is separate and remains governed by
https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE.

The `include`, `src`, `scripts`, and `tests` files are from that revision, except:

- `src/common.cpp` calls the model-scoped TFLLM graph bridge before allocation
  and around compute. It uses the existing pinned TFLLM ggml library, rather
  than linking a second ggml revision into the SDK.
- `src/codec_decoder.cpp` checks generated codebook indices before graph
  construction, so invalid generated codes return an error instead of aborting
  the process inside `ggml_get_rows`.
- `include/breeze/generation.h` and `src/generation.cpp` accept an optional
  caller-owned `GenPool`, retaining fused depth graphs and vocoder allocations
  between serialized requests while resetting per-request audio/KV state.
- `include/breeze/common.h`, `src/common.cpp`, `include/breeze/model.h`, and
  the stage entry points support scoped CPU thread counts. Inference on one
  engine is serialized; the previous thread count is restored on exceptions.
  The CPU backend owns a persistent ggml pool that grows to the largest stage
  thread count; idle workers sleep, and each graph uses its stage count.
- `include/breeze/gguf_loader.h` and `src/gguf_loader.cpp` provide selective
  CPU Q4_0/Q8_0 repacking, immutable vocoder transpose-convolution weight layouts,
  and Linux hugepage hints on owned CPU allocations. Raw NPU bridge sources
  remain in their original encoding; CPU repacking is skipped for NPU DD.
- `src/codec_decoder.cpp` reuses the model-owned immutable weight layouts.
  `tests/vocoder_stream.cpp` compares their output with the original path.
  An optional CPU SnakeBeta fusion preserves the original arithmetic; it is
  disabled by default and has not shown a material board speed improvement.

TFLLM-specific loading checks, NPU dispatch, C ABI, conversion policy, CLI and
HTTP service live outside this directory. The upstream HF tensor mappings,
Gemma normalization offset, frame generation, fused depth graph, CFG and
stateful vocoder are retained. `convert.py` uses the pinned scripts as a mapping
library and implements direct Q4/Q8 export and two-pass streaming writes.
`adapters/breeze/lookahead.cpp` reuses upstream `SpeechJob`/`step_jobs` for bounded,
ordered sentence batching with a shared initial voice reference. It runs on the
existing compute thread; upstream generation and its scheduler are unchanged.
`adapters/breeze/dialogue.cpp` instead overlaps single-stream code generation
with an independent CPU vocoder backend. Codec tensor metadata and mutable
stream state are private; immutable weight storage is shared with the model.
Its ordered worker callbacks finish before the serialized request returns.

The original PyTorch implementation was also inspected at
https://github.com/breezeblue-ai/breeze-tts/tree/a43d19e718b322e1cb3b2aa66f6d37d7945f4b21
for the nested Qwen3 backbone config, bidirectional T5Gemma2 attention, bundled
`audio_tokenizer/`, and per-stage fast execution boundaries. The exact reviewed
checkout is recorded in TFLLM's TTS documentation.

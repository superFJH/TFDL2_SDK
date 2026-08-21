# Qwen end-to-end A/B evaluation

Evaluation date: 2026-08-11

## Method

- Video: checkpoint `examples/soccer-broadcast.mp4`, 720 frames at 24 FPS.
- Codec input: all 32 official codec canvases, 4608 visual tokens.
- A: official `MageVLVisionPretrainedModel` output in BF16.
- B: `mage_vit_288x512.int8_fp16_topk2.fb` output, converted from FP16 to
  host float before Qwen injection.
- LLM: the same checkpoint, tokenizer, prompt, Qwen weights and greedy
  generation for A and B.
- Runtime: Transformers 5.15.0, BF16, NVIDIA RTX 5090. The repository requires
  Transformers 5.7 or newer. An initial unsupported 4.57 run also exposed an
  uninitialized non-persistent Qwen RoPE buffer in the meta-loaded lightweight
  bridge; that run was excluded, and the bridge has been fixed and retested.

The converted floating vision implementation has cosine `0.999573` against
the official visual class. INT8/FP16 Top-K=2 has cosine `0.817330` against the
official visual class over the complete 4608-token video.

## Directly verifiable factual questions

| Question | Official visual | INT8 visual | Result |
|---|---|---|---|
| Broadcaster logo on microphones | BBC Sport | BBC Sport | Exact |
| Presenters in the wide shot | Four | Four | Exact |
| Close-up before group shot | Yes | Yes | Exact |
| Presenter location | Large stadium, green pitch, red seats | Large stadium, green field, red seats | Same fact |

Both paths scored 4/4 against the extracted video frames. Three answers were
token-identical; the fourth differed only by `pitch` versus `field`.

## Open-ended questions

- Detailed description: both paths identify a gray-shirt presenter, yellow BBC
  Sport microphone, four-person group shot, packed stadium and discussion.
- Sport/action: both identify football/soccer and an ongoing match.
- Uniform color: both answer `white and blue uniforms`, token-identically.
- Chronological summary: both describe presenter close-up, transition to four
  presenters, stadium background and continued discussion; wording diverges.

Across all eight questions:

- initial Qwen logits cosine: `0.968802` mean;
- initial greedy token agreement: `7/8`;
- initial Top-10 overlap: `8.125/10` mean;
- exact complete answer tokens: `4/8`;
- token sequence ratio: `0.801683` mean;
- character sequence ratio: `0.824203` mean.

Exact-string agreement is intentionally lower for open-ended generation:
small logit changes choose different but semantically equivalent wording and
then autoregressive decoding follows a different path. On this one video no
material factual regression was observed at Top-K=2. This is a smoke test, not
a general VQA benchmark; a release decision should use a labeled multi-video
set and task-specific scoring.

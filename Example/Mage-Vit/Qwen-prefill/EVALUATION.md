# ARM H*S attention evaluation

Evaluation date: 2026-08-14. Host: 40-core Cortex-A77, no
`/dev/thinkforce*`; all layer FB results below use the software executor.

## Operator and graph checks

The standalone ARM Softmax cases cover scalar qinfo, direct H*S qinfo and
S=4/128/512. Against row dequantization followed by FP32 causal Softmax and
`[0,1]` UINT8 quantization, maximum error was 0 or 1 code. Every causal upper
triangle code equaled the output zero point.

For S=512, H=4 on Cortex-A77:

| Threads | Scalar speedup | H*S direct speedup |
|---:|---:|---:|
| 1 | 0.82x | 0.83x |
| 4 | 3.29x | 3.24x |
| 8 | 4.87x | 4.76x |
| 16 | 8.61x | 8.28x |
| 40 | 19.47x | 7.77x |

The baseline is the requested FP16 `Add(causal mask) -> Softmax -> Quantize`
chain. Single-thread ARM is slower; the S>=512 target of 1.5x is met from four
threads onward. These are warm software-executor measurements with two timed
iterations, not NPU/CPU handoff timings. The 40-thread result is visibly
scheduler-sensitive; 16 threads is the stable high-throughput point in this
run.

The layer audit confirms:

- three inputs (hidden/sin/cos), with no causal-mask placeholder;
- QK and scale-fold Requant both carry exactly H*S qinfo;
- scale-Requant qscale equals QK qscale `/sqrt(128)` within 2.1e-7 relative;
- QK -> scale-Requant codes are bit-exact, and the Custom input has H*S qinfo;
- probability and AV are scalar-qinfo UINT8;
- hidden/K/V outputs are FP16;
- no `PostQuant_`/`PostDeQuant_` nodes and no `Modify`-generated topology.

## FP16 RoPE fix and layer accuracy

Before the native FP16 `ApplyRope` dispatch, Qwen FP16 storage was read through
an unsupported path and layer-0 K-cache cosine was 0.0048. It is now 0.9992 to
0.9999 in the tested cases.

Four independent seq=4 synthetic prompts were unioned for calibration:

| Boundary | Layer 0 cosine | Layer 1 cosine |
|---|---:|---:|
| Q after RoPE | 0.997536 | 0.999332 |
| K after RoPE/export | 0.999225 | 0.999911 |
| QK | 0.991062 | 0.999642 |
| Probability | 0.993030 | 0.997099 |
| AV | 0.991465 | 0.996770 |
| Hidden output | 0.998154 | 0.995137 |
| V export | 0.998275 | 0.998423 |

For a real 869-token prompt assembled from one of the eight soccer video
visual shards, layer 0 produced hidden/K/V cosine
0.998653/0.999917/0.993475. H*S QK was better than scalar QK at every tested
attention boundary:

| Boundary | Scalar QK | H*S QK | H*S delta |
|---|---:|---:|---:|
| QK | 0.946586 | 0.947023 | +0.000437 |
| Probability | 0.900911 | 0.906652 | +0.005741 |
| AV | 0.930426 | 0.939072 | +0.008646 |

Of 27,808 H*S rows, 288 required range expansion to keep the gemmlowp
multiplier at or below 0.99.

## Full 36-layer Top-K sweep

The complete fixed-seq=4 stack, final RMSNorm, streamed LM head and cache
bundle were executed for Top-K=0/2/4/8. This is a synthetic regression bucket,
so it validates the machinery and drift behavior rather than VL quality.

| Top-K | Final hidden cosine | Logits cosine | Top-1 | Top-10 overlap |
|---:|---:|---:|:---:|---:|
| 0 | 0.851582 | 0.912614 | no | 4/10 |
| 2 | 0.984178 | 0.954067 | yes | 6/10 |
| 4 | 0.984146 | 0.954887 | yes | 7/10 |
| 8 | 0.983776 | 0.953774 | yes | 7/10 |

Top-K=2 selected `layers.6.mlp.down_proj` and
`layers.35.mlp.down_proj`, restoring the reference first token. Top-K=4 had
the best logits cosine and Top-10 overlap in this small bucket. The ordering is
not assumed to generalize; production Top-K must be ranked using the real
fixed-sequence prompt calibration set.

## Soccer S=898 Vit-style Tok hybrid

The official soccer video prompt contains 576 visual embeddings in an
898-token sequence. The prepared prompt automatically gives Tok-hybrid
boundaries `[0,21)`, `[21,887)` and `[887,898)`; the middle interval also
contains visual framing/separator tokens. In the stable baseline only
non-Top-K gated MLPs are split. H*S attention and UINT8 AV/o_proj retain their
established topology.

Layer-0 A/B on the same input showed that MLP-only splitting was the useful
part at the front of the network. Expanding the split to QKV changed K/V
cosine by less than `6e-6` and slightly reduced last-token cosine, so QKV
splitting is not enabled in the default preset.

| Layer-0 metric | Scalar baseline | MLP Tok hybrid |
|---|---:|---:|
| Hidden cosine | 0.998552 | 0.998796 |
| Last-token cosine | 0.966597 | 0.974033 |
| Text-token cosine | 0.960522 | 0.963978 |
| Visual-token cosine | 0.999399 | 0.999574 |

The complete 36-layer software run used the same H*S QK ranges and Top-K=2.

| Final metric | Scalar baseline | MLP Tok hybrid |
|---|---:|---:|
| Logits cosine | 0.89445 | 0.977850 |
| Top-1 agreement | yes | yes |
| Top-10 overlap | 4/10 | 8/10 |
| Software prefill time | 640.7 s | 631.6 s |

The same ORT W8A8 decoder consumed the Tok-hybrid FP16 K/V bundle directly,
generated 128 tokens at 8.51 token/s, and retained the same first token as the
FP32 prefill. The remaining large global hidden drift starts at Top-K layer 16
and is concentrated in visual token outliers; the final query token remains at
0.971983 cosine after layer 35, which explains why logits are substantially
better than the flattened hidden cosine of 0.619618 suggests.

### Selective QKV Tok hybrid and real-prompt Top-K

Later layers do not have layer 0's uniform QKV ranges. At layer 15, for
example, the K projection's prefix range is about `[-39.3, 32.2]`, while the
visual and final-query groups are about `[-4.35, 5.76]` and `[-5.28, 5.92]`.
Using exactly the same FP32 layer input, enabling QKV Tok hybrid at layer 15
produced the following controlled A/B:

| Layer-15 metric | MLP-only | MLP + QKV Tok hybrid |
|---|---:|---:|
| K export cosine | 0.987411 | 0.997261 |
| V export cosine | 0.980042 | 0.993057 |
| QK cosine | 0.996648 | 0.998569 |
| Probability cosine | 0.983851 | 0.988938 |
| AV cosine | 0.961037 | 0.963156 |
| Hidden cosine | 0.999728 | 0.999733 |

For the complete S=898 stack, layer 12 was the first selective QKV split.
Top-K=4 additionally placed the two next-largest real-prompt MLP branches,
layers 34 and 35, in FP16. All other source topology, ranges and weights were
held fixed.

| Final metric | MLP-only Top-K=2 | QKV@12 Top-K=2 | QKV@12 Top-K=4 |
|---|---:|---:|---:|
| Logits cosine | 0.977850 | 0.980857 | 0.984021 |
| Top-1 agreement | yes | yes | yes |
| Top-10 overlap | 8/10 | 7/10 | 7/10 |
| Layer-34 hidden cosine | 0.778543 | 0.787200 | 0.878862 |
| Layer-35 hidden cosine | 0.619618 | 0.665833 | 0.829907 |
| Software prefill time | 631.6 s | 630.5 s | 621.3 s |

The same ORT W8A8 decoder generated at 8.43--8.48 token/s for these caches.
All NPU-prefill variants retained the FP32 first token, but their greedy
generation diverged from the FP32-cache run at generated token index 2. Thus
the accuracy presets materially improve prefill logits and cache boundaries,
but do not yet justify a claim of lossless sequence generation.

## S=1024 deployment-prompt recheck (2026-08-20)

The superseded `[21,887]` deployment profile was not sequence-flexible in the semantic sense. Its
Tok-hybrid graph has static boundaries `[21,887]` and its four calibration
prompts have valid lengths `900/919/918/1024`.  Reassembling the returned
soccer frontend bundle with the deployment question `Describe this video.`
reproduces the target prompt exactly: valid length 781 and actual image-token
boundaries `[21,770]`.  It therefore puts the final query into the visual
activation group and also uses H*S QK rows below the calibrated length span.

A complete software execution of the deployed 36 FBs on that exact prompt
reproduced the NPU report.  Software/NPU logits cosine was
`0.77062788/0.77062848`; both selected Top-1 token `27` (`"<"`) versus reference
token `785` (`"The"`), with zero Top-10 overlap.  Software final-query hidden
cosine was only `0.720650`.  Layer-16 software/NPU flattened hidden cosine was
`0.71327333/0.71327367`, and layer 35 was `0.71546857/0.71546850`; the printed
K/V metrics also agree at the displayed precision.  Consequently the complete
accuracy failure is reproduced without NPU execution and the layer-16 drop is
an FB/calibration behavior, not an NPU-only numerical failure.

The NPU compiler separately warns that grouped QK MatMul output qinfo does not
cover an accepted row layout.  That warning still needs a compile-coverage and
latency fix, but it does not explain this accuracy failure because the software
path reproduces the layer curve.  A deployable flexible-video profile must not
use the calibration prompt's final-image position as a fixed graph split; at
minimum it needs representative short prompts and a stable prefix-versus-rest
grouping, followed by a fresh full-stack task-accuracy check.

A controlled layer-0 rebuild used the real 781-token prompt plus one real
full-length 1024-token prompt for calibration.  Changing no projection or
attention topology gave:

| Tok-hybrid boundaries | Hidden cosine | Final-query cosine |
|---|---:|---:|
| deployed `[21,887]` | 0.997244 | 0.930697 |
| exact prompt `[21,770]` | 0.998993 | 0.979384 |
| stable prefix/rest `[21]` | 0.998915 | 0.976932 |

The coarse `[21]` split retains almost all of the exact prompt-specific gain
without baking a video's last-image position into graph shapes.  It is the
recommended starting topology for the next flexible-video 36-layer rebuild;
its ranges must still aggregate multiple videos, question lengths and a real
full-length prompt before deployment.

An experimental full 36-layer `[21]`, QKV@12, Top-K=4 stack was then calibrated
with the real 781-token prompt and one real full-length 1024-token prompt.  On
the soccer prompt, final-query cosine improved from `0.720650` to `0.948047`,
logits cosine from `0.770628` to `0.955072`, Top-1 changed from token 27 back to
the reference token 785, and Top-10 overlap improved from 0/10 to 7/10.  The
existing ORT W8A8 decoder generated a coherent description beginning:

> The video opens with a man in a black shirt holding a yellow microphone with
> "BBC Sport" written on it.

It decoded 64 tokens at 8.32 token/s with 16 CPU threads.  At the user's
request these artifacts replace the old model under the deployment field-test
profile name `s1024-flex-topk4-qkv12`. The old `[21,887]` artifacts have been
removed. This is still not a production calibration set: results from new
videos, English/Chinese questions and different prompt lengths must be used to
decide whether more representative ranges are required.

ORT W8A8 is a materially easier quantization problem: its decode MatMul uses a
fresh per-token activation scale at every step, while normalization, RoPE,
attention, residuals and KV storage stay floating. The NPU prefill must use
static calibration scales for an 898-token mixture of text, visual and query
roles. The measured decoder result is therefore not evidence that static NPU
W8A8 should have identical error.

### S=1024 Top-K=0 speed A/B artifact

A second 36-layer field-test stack, `s1024-flex-topk0-qkv12`, was exported
from exactly the same `[21]` two-prompt ranges as the Top-K=4 stack. HxS QK,
ARM causal Softmax and QKV Tok hybrid from layer 12 are unchanged; only the
global `o_proj/down_proj` FP16 bypass count changes from four to zero. Its
layer 6/16/34/35 FBs shrink from about 176.9 MB to about 102.5 MB and all 36
INT8 tensor audits pass. Hardware task accuracy is intentionally left for the
deployment-machine A/B.

### QKV Tok-hybrid channel-major layout rebuild

Both deploy profiles were rebuilt with the optimized QKV layout from layer 12.
Each token group now remains channel-major through Q/K/V projection; groups are
concatenated on the token axis before the three required full-sequence BHSD
transposes. For the INT8 layer 12 graph this reduces Transpose nodes from 18 to
11, Reshape nodes from 23 to 20 and software commands from 98 to 88. On the real
781-token soccer prompt, the rebuilt layer 12 hidden, K and V outputs are all
bit-exact to the previous layout. All 72 tensor audits across the Top-K=0 and
Top-K=4 stacks pass with zero invalid INT8 qinfo. Target NPU latency remains the
deployment-machine measurement.

## Remaining hardware evaluation

The complete 32-canvas soccer prompt was assembled successfully at S=6527.
Its full 36-layer software run was not used as a performance result: QK alone
has about 1.36 billion score elements per layer, and no NPU device is present.
The runner, cache bundle and `decode_from_prefill_cache.py` handoff are in
place. Full-video first-token/generation and real QK/AV latency should be run
when `/dev/thinkforce*` is available; generation also requires the separate
`transformers>=5.7` Qwen environment.

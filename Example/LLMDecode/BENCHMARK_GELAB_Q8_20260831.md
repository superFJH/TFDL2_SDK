# GELab Q8 GGUF decode benchmark

Date: 2026-08-31

## Setup

- CPU: 40-core Cortex-A77, ARM DOTPROD available.
- Build: `Release`, `GGML_CPU_KLEIDIAI=ON`, `GGML_NATIVE=ON`.
- Decoder: the local `GELab-Zero-4B-preview` checkpoint converted directly to
  Q8_0 GGUF (3.98 GiB / 8.50 BPW).
- External prefix: existing GELab TFDL job with 173 valid tokens, 36 layers,
  8 KV heads and head dimension 128.
- Generation: 128 single-token decode calls; throughput excludes model mmap,
  context setup and external-KV import.  The runner used 32 threads unless
  stated otherwise.

## Results

| GGUF weights | KV cache | Threads | KV import | Decode | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| Q8_0 | FP16 | 32 | 7.1 ms | 11.906 s | **10.751 tok/s** |
| Q8_0 | Q8_0 | 32 | 49.6 ms | 12.409 s | 10.315 tok/s |
| Q8_0 | Q8_0 | 40 | 49.5 ms | 15.885 s | 8.058 tok/s |

For comparison, the existing ONNX decode measurement supplied for this
development machine is 10.5 tok/s.  The Q8_0-weight + FP16-KV configuration
is therefore the currently verified candidate; 32 threads should be used on
this machine, not all 40 cores.

## Accuracy status

The Q8_0 KV-cache run diverged from the FP16 KV-cache run at the first token
computed by the CPU decoder.  The external Q8 importer is functional and the
cache takes the expected FlashAttention layout, but this accuracy regression
must be resolved before enabling Q8 KV in production.  Consequently the
project defaults to `--kv-cache-type fp16`; Q8 KV remains an explicit
experimental option.

The Q8_0 GGUF itself is a weight-only on-disk format.  llama.cpp/KleidiAI
performs activation packing/quantization in its selected ARM kernels at
runtime; GGUF does not encode ONNX-style static activation QDQ nodes.

## DRAM bandwidth check

`tools/memory_bandwidth.cpp` reads a 4 GiB buffer four times with contiguous,
per-worker shards.  This is deliberately larger than the 48 MiB shared L3 and
models the decoder's repeated traversal of GGUF tensors.

| Threads | Sustained read bandwidth |
| ---: | ---: |
| 1 | 15.3 GB/s |
| 8 | 54.7 GB/s |
| 16 | 55.4 GB/s |
| 32 | **51.3 GB/s** |
| 40 | 49.7 GB/s |

At 10.751 tok/s, traversing the 3.98 GiB Q8 GGUF once per token implies about
42.8 GiB/s (46.0 GB/s) of model-weight traffic.  That is approximately 90% of
the 32-thread streaming-read result.  Therefore steady-state decode is
DRAM-bandwidth limited on this host, rather than limited by SSD/storage I/O.
The drop at 40 threads matches bandwidth saturation/oversubscription; use 32
threads.  Hardware counter collection was unavailable because this host sets
`kernel.perf_event_paranoid=4`; the result should be rechecked with PMU access
enabled if a precise DRAM-controller read counter is required.

## Q4_K_M direct weight-only experiment

The Q4 model was quantized from a fresh F16 GGUF exported directly from the
original GELab checkpoint; it was **not** requantized from the Q8 model.  The
`llmquantize` wrapper calls llama.cpp's public block-wise quantization API,
which needs no activation calibration data.  Q4_K_M is a mixed format: most
matrices are Q4_K, while selected sensitive matrices use Q6_K.  Its reported
size is 2,375.91 MiB / 4.95 BPW.

| GGUF weights | Model size | Decode throughput | Relative to Q8 |
| --- | ---: | ---: | ---: |
| F16 | 7.49 GiB | 5.800 tok/s | 0.54x |
| Q8_0 | 3.98 GiB | 10.755 tok/s | 1.00x |
| Q4_K_M | 2.32 GiB | **16.084 tok/s** | **1.50x** |

All measurements use the same 173-token external FP16 KV prefix, 128-token
limit and 32 decode threads.  The F16 response emitted EOS after 121 decode
calls; Q8 and Q4 reached the 128-token cap.  The Q4 result is therefore a
49.5% throughput improvement over Q8 while reducing GGUF storage by 41.7%.

For this UI-image case, F16, Q8 and Q4 all identified the Xiaomi **Apps**
settings page and described the same seven visible functions (system app
settings, manage apps, shortcuts, dual apps, permissions, app lock and Mi
Protect).  Greedy Q8 and Q4 paths first diverged from F16 at token 11; Q4 and
Q8 agreed with one another through token 16.  This is expected from close
logit rankings and does not by itself indicate an incorrect UI action.  The
semantic result is acceptable for this one case, but it is not a substitute
for a multi-screen action-accuracy evaluation.

## Q4_K DOTPROD kernel investigation

The selected Cortex-A77 path is llama.cpp's `Q4_K_8x4 × Q8_K` NEON DOTPROD
kernel, not a KleidiAI Q4_K kernel.  KleidiAI currently accelerates Q4_0 and
Q8_0 only.  Its 6-bit Q4_K scale/min unpacking is already inlined by GCC.

An experimental load-time expansion of those 6-bit values was rejected: it
kept all 128 generated token IDs identical but changed the Q4_K metadata
traffic from 96 to 128 bytes per 8-column superblock (+33% metadata, about
+11% Q4_K packed-weight traffic) and produced no single-stream speedup.

The useful no-format-change optimization is LTO.  With a matched GCC 11 C/C++
toolchain, 32 threads and the same external FP16 KV prefix, two alternating
128-token runs produced:

| Build | Decode throughput | Token IDs |
| --- | ---: | --- |
| Release | 16.05 tok/s mean | baseline |
| Release + LTO | **16.43 tok/s mean** | all 128 identical |

That is a measured **2.3%** improvement. `scripts/build_arm.sh` now enables
LTO by default and automatically selects a matching GCC pair when possible.
Use `LLMDECODE_LTO=OFF` to disable it.  Larger gains require a lower-byte
weight format (or hardware with I8MM/SVE), not another Q4_K scale-unpack path.

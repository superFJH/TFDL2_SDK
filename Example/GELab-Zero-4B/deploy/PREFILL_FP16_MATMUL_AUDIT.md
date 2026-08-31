# GELab S1024 FP16 grouped-GQA BSC MatMul audit

The 36 prefill layer FBs in
`model/prefill/s1024-fp16-masksoftmax-grouped-gqa` use a BSC `[B,S,C]` main trunk.
Each layer contains seven FP16 projection MatMuls:

- Q/K/V/O
- Gate/Up/Down

Every projection has the form `A[B,S,K] @ W[K,N]`. The activation is input 0,
the pre-transposed constant weight is input 1, and `transA=false` plus
`transB=false`. Attention alone temporarily changes from BSC to BHSD for RoPE,
QK, MaskSoftmax and AV, then returns to BSC before O projection and the
residual add. The checkpoint has 32 Q heads and 8 KV heads. Each grouped
MatMul folds the four query heads associated with one KV head into its M axis;
K/V are never materialized with `repeat_interleave` or `Concat`.

## Exported graph audit

- Layer FBs: 36 (`layer_00` through `layer_35`)
- Projection audits: 36
- Audited projections: 252
- Valid projections: 252
- Weight dtype set: `TFDataType.TFDL_FLOAT16`
- Operator set: `MatMul`
- Main-trunk layout set: `BSC`
- Activation-left checks: 252/252
- Weight-right checks: 252/252
- `transA=false`: 252/252
- `transB=false`: 252/252
- Grouped-GQA audit: 36/36 layers valid; `Hq=32`, `Hkv=8`, `repeat=4`
- Materialized K/V-repeat operators: 0

Each layer's complete machine-readable result is stored beside the FB as
`layer_XX.matmul-audit.json`.

## S1024 semantic audit

Layer 0 was compared against the original PyTorch graph with seed 1234 and a
full 1024-token input. PyTorch ran on CUDA and the FB ran on the TFDL software
executor. Selected cosine results are:

| Tensor | Global cosine | Minimum token cosine |
| --- | ---: | ---: |
| Q projection | 0.9999999142 | 0.9999997269 |
| K projection | 0.9999999143 | 0.9999997048 |
| V projection | 0.9999999140 | 0.9999998746 |
| QK | 0.9999996666 | n/a |
| MaskSoftmax probability | 0.9999983607 | n/a |
| AV | 0.9999983218 | 0.9999953058 |
| O projection | 0.9999987972 | 0.9999964350 |
| Gate projection | 0.9999997738 | 0.9999995633 |
| Up projection | 0.9999997957 | 0.9999996307 |
| Down projection | 0.9999997278 | 0.9999994351 |
| Final hidden output | 0.9999997668 | 0.9999995377 |

The semantic audit also found three runtime inputs, no external causal-mask
input, no quantization tensors, FP16 probability/AV/hidden/K/V outputs, and a
fully zero causal upper triangle after native `MaskSoftmax`.

#pragma once
#include "tfllm/engine.h"

namespace tfllm {
// Contiguous [B,H,L,D] buffers borrowed until every head has drained. Output
// must not alias any input. GQA groups adjacent query heads around one KV head.
struct DenseAttentionShape {
    size_t batch,query_heads,kv_heads,queries,keys,query_dim,value_dim;
};
// Optional B/H/row strides for Q,K,V,output; contiguous channel axes.
// Strided dense execution currently requires equal query/KV head counts.
struct DenseAttentionStrides {size_t values[4][3];};
// Borrowed broadcasted [B,H,Q,K] mask; strides are in elements. Boolean true
// is visible; floating values are additive logits (finite or -infinity).
struct DenseAttentionBias {const void* data=nullptr;size_t strides[4]={};bool boolean=false;};
// Common FAST inner kernel. No nested head-pool submission. Logical QK/value
// channels must be <=256; native U8 scores retain saturated-row I32 retries.
void RunFastAttentionHead(Linear&,Fp32InputView,Fp32InputView,Fp32InputView,Fp32OutputView,
                          AttentionMask,const std::shared_ptr<AttentionWorkspaceScope>& = {});
void RunFastAttentionHead(Linear&,Fp16InputView,Fp16InputView,Fp16InputView,Fp16OutputView,
                          AttentionMask,const std::shared_ptr<AttentionWorkspaceScope>& = {});
void RunFastWindowAttention(Linear&,const float*,const float*,const float*,float*,
                           size_t jobs,size_t queries,size_t keys,size_t dim,size_t heads,float,const size_t*);
void RunFastWindowAttention(Linear&,const uint16_t*,const uint16_t*,const uint16_t*,uint16_t*,
                           size_t jobs,size_t queries,size_t keys,size_t dim,size_t heads,float,const size_t*);
// Strict fused path: no dense score/probability tensor and no CPU fallback.
// Logical channels/keys may have tails; physical padding stays internal.
// Causal uses SDPA upper-left alignment even when queries != keys.
// fast selects centered/orthogonal precision transforms, U8 QK and FP32
// softmax scales for all supported shapes, causal/noncausal and intervals.
void RunFusedAttentionFp16(Linear&,const uint16_t* q,const uint16_t* k,const uint16_t* v,
                          uint16_t* output,const DenseAttentionShape&,bool causal,float scale,
                          const int64_t* intervals=nullptr,const DenseAttentionStrides* strides=nullptr,bool fast=false,const DenseAttentionBias* bias=nullptr);
// Torch external FP32 entry; independent of serving's FP16 boundaries.
// fast uses the common FAST kernel; QK/value channels must be <=256.
// Explicit fast or the backend option enables it, independently of model ID.
void RunFusedAttentionFp32(Linear&,const float*,const float*,const float*,float*,
                          const DenseAttentionShape&,bool causal,float scale,const int64_t* intervals=nullptr,
                          const DenseAttentionStrides* strides=nullptr,bool fast=false,const DenseAttentionBias* bias=nullptr);
// Native FP16 prefill uses canonical KV and indexed interleaved query/output.
// False is a pre-submit profile/capability rejection; device errors propagate.
bool TryFastCachedAttentionFp16(Linear&,const Config&,const KvCache&,size_t layer,
                                const Execution&,const Activation&,Activation&);
// Optional contiguous intervals [batch, queries, 2] override causal. They are
// shared across heads, use logical key indices, and may be empty (zero output).

// CSR rows flatten [batch, query_head, ceil(queries/query_block_size)].
// Columns are logical KV block IDs, strictly increasing within each row.
// Every query in a block sees the same selected keys. GQA still maps each
// query head to its KV head; masks are NOT shared across query heads.
struct BlockSparseAttentionMask {
    const int64_t* row_offsets=nullptr;
    size_t offsets_count=0;
    const int64_t* column_indices=nullptr;
    size_t indices_count=0;
    size_t query_block_size=128,kv_block_size=128;
};
// Native gather -> one fused attention per CSR row. Empty rows return zero.
// Global key length may exceed 16384, but selected logical keys per row must
// not exceed 16384. Query blocks 1..1024, KV blocks 1..16384. No CPU fallback.
// CSR is snapshotted/validated before output writes or device submission.
// Q/K/V remain borrowed and must be immutable until all accepted jobs drain.
void RunBlockSparseAttentionFp16(Linear&,const uint16_t* q,const uint16_t* k,const uint16_t* v,
                               uint16_t* output,const DenseAttentionShape&,
                               const BlockSparseAttentionMask&,float scale);
void RunBlockSparseAttentionFp32(Linear&,const float*,const float*,const float*,float*,
                               const DenseAttentionShape&,const BlockSparseAttentionMask&,float scale);
}

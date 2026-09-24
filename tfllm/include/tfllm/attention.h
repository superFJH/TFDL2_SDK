#pragma once
#include "tfllm/engine.h"

namespace tfllm {
// Contiguous [B,H,L,D] buffers borrowed until every head has drained. Output
// must not alias any input. GQA groups adjacent query heads around one KV head.
struct DenseAttentionShape {
    size_t batch,query_heads,kv_heads,queries,keys,query_dim,value_dim;
};
// Strict fused path: no dense score/probability tensor and no CPU fallback.
// Logical channels/keys may have tails; physical padding stays internal.
// Causal uses SDPA upper-left alignment even when queries != keys.
void RunFusedAttentionFp16(Linear&,const uint16_t* q,const uint16_t* k,const uint16_t* v,
                          uint16_t* output,const DenseAttentionShape&,bool causal,float scale,
                          const int64_t* intervals=nullptr);
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
}

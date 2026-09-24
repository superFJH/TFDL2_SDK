#pragma once
#include "tfllm/model.h"
#include <functional>

namespace tfllm {
// IEEE binary16 bits, never a vector<float> with rounded values.
using Activation = std::vector<uint16_t>;
Activation HalfActivation(const std::vector<float>&);
std::vector<float> FloatActivation(const Activation&);
namespace cpu {
const char* KernelName();
// Shared persistent CPU workers. Call before pinning NPU worker threads.
void Initialize();
// Grow the process-wide pool to at least max_workers-1 background threads.
// Explicit 1..32 participants, not capped by hardware_concurrency. Growth is
// monotonic; configure before pinning NPU threads. Fresh processes for A/B tests.
void Initialize(size_t max_workers);
size_t BackgroundWorkers();
// Actual callback count for this shape/limit (0 for empty, 1 for small work).
size_t RowParticipants(size_t rows,size_t width,size_t max_workers);
// Only CPU kernels belong here: callbacks may not submit/wait for NPU work.
void Rows(size_t rows,size_t width,const std::function<void(size_t,size_t)>&);
// Bound participants INCLUDING the caller. Useful when several NPU lanes
// already run CPU kernels concurrently; 1 stays entirely on the caller.
void RowsLimited(size_t rows,size_t width,size_t max_workers,const std::function<void(size_t,size_t)>&);
void Narrow(const float*,uint16_t*,size_t);
void Widen(const uint16_t*,float*,size_t);
void EmbeddingRow(const Tensor&,size_t token,uint16_t*,float scale);
float QuantizeRow(const uint16_t*,uint8_t*,size_t);
// Out-of-place byte transpose: contiguous [rows][columns] -> [columns][rows].
// No arithmetic/quantization; supports unaligned pointers and tail dimensions.
void TransposeU8(const uint8_t* src,uint8_t* dst,size_t rows,size_t columns);
// Bitwise out-of-place FP16 transpose with strides measured in uint16_t.
// Source [rows][columns] -> destination [columns][rows]; padding is untouched.
void TransposeFp16(const uint16_t* src,uint16_t* dst,size_t rows,size_t columns,
                   size_t src_stride,size_t dst_stride);
void DequantizeRow(const int32_t*,uint16_t*,size_t,float,const float*);
void AccumulateRow(const int32_t*,float*,size_t,float,const float*);
void Norm(const uint16_t*,uint16_t*,size_t rows,size_t width,const float* weight,float epsilon);
void LayerNorm(const uint16_t*,uint16_t*,size_t rows,size_t width,const float* weight,const float* bias,float epsilon);
void Gelu(uint16_t*,size_t,bool exact=false);
void Add(uint16_t*,const uint16_t*,size_t);
// Projection BiasAdd is an FP16 boundary: both operands and the result are
// binary16. Norm affine parameters remain FP32 and use LayerNorm/Norm above.
void Bias(uint16_t*,size_t rows,size_t width,const uint16_t*);
void Gate(uint16_t*,const uint16_t*,size_t,bool gelu);
void SigmoidMul(uint16_t* value,const uint16_t* gate,size_t);
void Rope(uint16_t*,size_t rows,size_t heads,size_t dim,bool neox,const float* cosine,const float* sine);
// FP16 scores/probabilities; only one row of FP32 exp scratch per CPU worker.
void Softmax(uint16_t*,size_t columns,size_t first,size_t valid,float scale,float cap,std::vector<float>& scratch);
// Bit-equivalent to Softmax -> QuantizeRow, including FP16 probability rounding.
float SoftmaxQuantize(const uint16_t*,uint8_t*,size_t columns,size_t first,size_t valid,float scale,float cap,std::vector<float>& scratch);
struct Int32ScoreSpan {const int32_t* values;const float* scales;size_t columns;};
// Ordered column shards for ONE row. Only row-local FP32 scratch is written;
// no FP16 logits/probability tile. Preserves DequantizeRow -> SoftmaxQuantize.
float SoftmaxQuantizeI32(const Int32ScoreSpan*,size_t spans,uint8_t*,size_t columns,
                        size_t first,size_t valid,float query_scale,float scale,float cap,std::vector<float>& scratch);
}
}

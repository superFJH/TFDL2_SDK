#pragma once
#include "tfllm/activation.h"
#include <array>

namespace tfllm {
// Native spatial path, NCHW/OIHW. Pointwise convolution uses Linear instead.
struct SpatialConvSpec {
    size_t input_channels=0, output_channels=0, kh=0, kw=0, stride=1;
    std::array<size_t,4> padding{}; // top, bottom, left, right
    void Validate() const;
    std::array<size_t,2> Output(size_t height,size_t width) const;
};
struct AffineU8 {float scale=1; uint8_t zero=0;};
AffineU8 AffineRange(const uint16_t* data,size_t count);
void QuantizeAffine(const uint16_t*,uint8_t*,size_t,AffineU8);
void DequantizeSpatial(const int32_t*,uint16_t*,size_t,float scale);
struct SpatialConvWeight {
    SpatialConvSpec spec;
    std::vector<uint8_t> bytes;
    std::vector<float> scales;
    std::vector<uint8_t> zeros;
    virtual ~SpatialConvWeight()=default;
};
std::shared_ptr<SpatialConvWeight> QuantizeSpatialWeight(const SpatialConvSpec&,const uint16_t*);
// Explicit host integer oracle, not an implicit NPU fallback.
void ReferenceSpatialConv(const SpatialConvWeight&,const uint16_t*,uint16_t*,size_t batch,size_t height,size_t width);
}

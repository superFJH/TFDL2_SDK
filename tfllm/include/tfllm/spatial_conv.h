#pragma once
#include "tfllm/activation.h"
#include <array>

namespace tfllm {
// The SDK splits large Conv reductions internally. Bound the complete signed
// accumulator, rather than imposing its 4608-term per-command limit on a layer.
inline constexpr size_t SpatialConvMaxKernel=size_t(INT32_MAX)/(255u*255u);
// Native spatial/pointwise path, NCHW/OIHW.
struct SpatialConvSpec {
    size_t input_channels=0, output_channels=0, kh=0, kw=0, stride=1;
    std::array<size_t,4> padding{}; // top, bottom, left, right
    bool transposed=false; // Stride 2; SDK OIHW weights have reversed spatial taps.
    std::array<size_t,2> output_padding{};
    void Validate() const;
    std::array<size_t,2> Output(size_t height,size_t width) const;
};
struct AffineU8 {float scale=1; uint8_t zero=0;};
AffineU8 AffineRange(const uint16_t* data,size_t count);
AffineU8 AffineRange(const float* data,size_t count);
void QuantizeAffine(const uint16_t*,uint8_t*,size_t,AffineU8);
void QuantizeAffine(const float*,uint8_t*,size_t,AffineU8);
// Quantize contiguous NCHW channel planes; leave destination padding untouched.
void QuantizeAffinePlanar(const uint16_t*,uint8_t*,size_t channels,size_t image,size_t stride,AffineU8);
void QuantizeAffinePlanar(const float*,uint8_t*,size_t channels,size_t image,size_t stride,AffineU8);
void DequantizeSpatial(const int32_t*,uint16_t*,size_t,float scale);
void DequantizeSpatial(const int32_t*,float*,size_t,float scale);
// Round channel bias into the INT32 accumulator domain; reject nonfinite
// values and reserve headroom for every possible UINT8 convolution dot.
std::vector<int32_t> QuantizeSpatialBias(const uint16_t*,size_t channels,size_t kernel,
                                       float input_scale,const float* weight_scales);
struct SpatialConvWeight {
    SpatialConvSpec spec;
    std::vector<uint8_t> bytes;
    std::vector<float> scales;
    std::vector<uint8_t> zeros;
    bool pointwise_symmetric=false;
    virtual ~SpatialConvWeight()=default;
};
std::shared_ptr<SpatialConvWeight> QuantizeSpatialWeight(const SpatialConvSpec&,const uint16_t*);
std::shared_ptr<SpatialConvWeight> QuantizeSpatialWeight(const SpatialConvSpec&,const float*);
std::shared_ptr<SpatialConvWeight> QuantizePointwiseWeight(const SpatialConvSpec&,const uint16_t*);
// Explicit host integer oracle, not an implicit NPU fallback.
void ReferenceSpatialConv(const SpatialConvWeight&,const uint16_t*,uint16_t*,size_t batch,size_t height,size_t width,
                          const uint16_t* bias=nullptr);
void ReferenceSpatialConvFp32(const SpatialConvWeight&,const float*,float*,size_t batch,size_t height,size_t width);
}

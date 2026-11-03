// Squeeze-and-excitation block with floating point computation and optional
// UINT8 boundaries. Inputs: NCHW activation, FC1 weight/bias, FC2 weight/bias.
// Weights are FP32 with shapes [hidden, channels] and [channels, hidden].

#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

using namespace TFDL_CAPI;

namespace TFDLOP {
namespace SEBlock {

struct Param {
    std::string outputType = "auto";
    int batches = 0;
    int channels = 0;
    int hidden = 0;
    int spatial = 0;
    int threads = 1;
    bool fastQuant = true;
#if defined(__aarch64__)
    bool fp16Weights = false;
    std::once_flag packOnce;
    std::vector<__fp16> w1Half;
    std::vector<__fp16> w2Half;
#endif
};

static int WorkerCount() {
    const char *value = std::getenv("TFDL_SEBLOCK_THREADS");
    if (value == nullptr || *value == '\0') return 1;
    char *end = nullptr;
    const long count = std::strtol(value, &end, 10);
    if (*end != '\0' || count < 1 || count > 16)
        throw std::runtime_error("TFDL_SEBLOCK_THREADS must be in [1,16]");
    return (int)count;
}

static TFTensor TensorOrParam(TFContext context, const std::string &name) {
    auto value = GetTensorByName(context, name);
    if (!value.IsValid()) value = GetParam(context, name);
    return value;
}

static void Prepare(TFContext, TFNode node) {
    std::string error;
    const auto config = json11::Json::parse(GetNodeCustomJsonStr(node), error);
    if (!error.empty()) throw std::runtime_error("SEBlock invalid JSON: " + error);
    const int threads = WorkerCount();
    auto *param = new Param();
    param->threads = threads;
    const char *fastQuant = std::getenv("TFDL_SEBLOCK_FAST_QUANT");
    param->fastQuant = fastQuant == nullptr || std::string(fastQuant) != "0";
#if defined(__aarch64__)
    const char *fp16Weights = std::getenv("TFDL_SEBLOCK_FP16_WEIGHTS");
    param->fp16Weights = fp16Weights != nullptr && std::string(fp16Weights) == "1";
#endif
    const auto mode = config["outputType"].string_value();
    if (!mode.empty()) param->outputType = mode;
    if (param->outputType != "auto" && param->outputType != "float32"
            && param->outputType != "uint8") {
        delete param;
        throw std::runtime_error("SEBlock outputType must be auto, float32, or uint8");
    }
    FreeNodeCustomParam(node, [](void *value) { delete (Param *)value; });
    NewNodeCustomParam(node, [param]() -> void * { return param; });
}

static void Reshape(TFContext context, TFNode node) {
    const auto info = GetNodeInfo(node);
    if (info.InputNames.size() != 5 || info.OutputNames.size() != 1)
        throw std::runtime_error("SEBlock requires activation and four FC parameters");
    const auto input = GetTensorByName(context, info.InputNames[0]);
    const auto output = GetTensorByName(context, info.OutputNames[0]);
    const auto shape = GetTensorShape(input);
    if (shape.size() != 4 || shape[0] < 1 || shape[1] < 1
            || shape[2] < 1 || shape[3] < 1)
        throw std::runtime_error("SEBlock requires positive NCHW dimensions");
    const int channels = shape[1];
    auto w1 = TensorOrParam(context, info.InputNames[1]);
    auto b1 = TensorOrParam(context, info.InputNames[2]);
    auto w2 = TensorOrParam(context, info.InputNames[3]);
    auto b2 = TensorOrParam(context, info.InputNames[4]);
    if (!w1.IsValid() || !b1.IsValid() || !w2.IsValid() || !b2.IsValid())
        throw std::runtime_error("SEBlock missing FC parameter");
    const auto firstShape = GetTensorShape(w1);
    if (firstShape.size() != 2 || firstShape[0] < 1
            || firstShape[1] != channels)
        throw std::runtime_error("SEBlock FC1 weight must have shape [hidden, channels]");
    const int hidden = firstShape[0];
    if (GetTensorShape(b1) != std::vector<int>{hidden}
            || GetTensorShape(w2) != std::vector<int>({channels, hidden})
            || GetTensorShape(b2) != std::vector<int>{channels})
        throw std::runtime_error("SEBlock FC parameter dimensions disagree");
    for (auto parameter : {w1, b1, w2, b2}) {
        if (GetTensorType(parameter) != TFCAPI_FLOAT)
            throw std::runtime_error("SEBlock FC parameters must remain FP32");
    }
    const auto inputType = GetTensorType(input);
    if (inputType != TFCAPI_FLOAT && inputType != TFCAPI_UINT8)
        throw std::runtime_error("SEBlock input must be FP32 or UINT8");
    const auto *param = (const Param *)GetNodeCustomParam(node);
    if (param == nullptr) throw std::runtime_error("SEBlock parameter state is missing");
    const auto outputType = param->outputType == "auto" ? inputType
        : param->outputType == "float32" ? TFCAPI_FLOAT : TFCAPI_UINT8;
    ReSizeTensor(output, shape);
    SetTensorType(output, outputType);
    auto *writable = (Param *)param;
    writable->batches = shape[0];
    writable->channels = channels;
    writable->hidden = hidden;
    writable->spatial = shape[2] * shape[3];
}

static inline float Dot(const float *a, const float *b, int count) {
#if defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= count; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    const float32x4_t pair = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    float sum = vaddvq_f32(pair);
    for (; i < count; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

#if defined(__aarch64__)
static inline float DotHalf(const float *a, const __fp16 *b, int count) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= count; i += 16) {
        const float16x8_t first = vld1q_f16((const float16_t *)(b + i));
        const float16x8_t second = vld1q_f16((const float16_t *)(b + i + 8));
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vcvt_f32_f16(vget_low_f16(first)));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vcvt_f32_f16(vget_high_f16(first)));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vcvt_f32_f16(vget_low_f16(second)));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vcvt_f32_f16(vget_high_f16(second)));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < count; ++i) sum += a[i] * (float)b[i];
    return sum;
}
#endif

static inline float SumFloat(const float *data, int count) {
#if defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= count; i += 16) {
        s0 = vaddq_f32(s0, vld1q_f32(data + i));
        s1 = vaddq_f32(s1, vld1q_f32(data + i + 4));
        s2 = vaddq_f32(s2, vld1q_f32(data + i + 8));
        s3 = vaddq_f32(s3, vld1q_f32(data + i + 12));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < count; ++i) sum += data[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += data[i];
    return sum;
#endif
}

static inline uint64_t SumBytes(const uint8_t *data, int count) {
    uint64_t sum = 0;
    int i = 0;
#if defined(__aarch64__)
    for (; i + 16 <= count; i += 16) sum += vaddlvq_u8(vld1q_u8(data + i));
#endif
    for (; i < count; ++i) sum += data[i];
    return sum;
}

static inline void MultiplyFloat(float *dst, const float *src, float gate, int count) {
    int i = 0;
#if defined(__aarch64__)
    for (; i + 16 <= count; i += 16) {
        vst1q_f32(dst + i, vmulq_n_f32(vld1q_f32(src + i), gate));
        vst1q_f32(dst + i + 4, vmulq_n_f32(vld1q_f32(src + i + 4), gate));
        vst1q_f32(dst + i + 8, vmulq_n_f32(vld1q_f32(src + i + 8), gate));
        vst1q_f32(dst + i + 12, vmulq_n_f32(vld1q_f32(src + i + 12), gate));
    }
#endif
    for (; i < count; ++i) dst[i] = src[i] * gate;
}

static inline void MultiplyBytesToFloat(float *dst, const uint8_t *src,
                                       float scale, float zero, float gate, int count) {
    int i = 0;
#if defined(__aarch64__)
    const float32x4_t zeros = vdupq_n_f32(zero);
    for (; i + 16 <= count; i += 16) {
        const uint8x16_t bytes = vld1q_u8(src + i);
        const uint16x8_t low = vmovl_u8(vget_low_u8(bytes));
        const uint16x8_t high = vmovl_u8(vget_high_u8(bytes));
        const uint32x4_t ints[4] = {
            vmovl_u16(vget_low_u16(low)), vmovl_u16(vget_high_u16(low)),
            vmovl_u16(vget_low_u16(high)), vmovl_u16(vget_high_u16(high))};
        for (int j = 0; j < 4; ++j) {
            float32x4_t values = vsubq_f32(vcvtq_f32_u32(ints[j]), zeros);
            values = vmulq_n_f32(values, scale);
            vst1q_f32(dst + i + 4 * j, vmulq_n_f32(values, gate));
        }
    }
#endif
    for (; i < count; ++i) dst[i] = ((float)src[i] - zero) * scale * gate;
}

static inline void MultiplyBytesToQuant(uint8_t *dst, const uint8_t *src,
                                       float inputScale, float inputZero, float gate,
                                       float outputScale, float outputZero, int count) {
    const float invOutputScale = 1.0f / outputScale;
    int i = 0;
#if defined(__aarch64__)
    const float32x4_t zeros = vdupq_n_f32(inputZero);
    const float32x4_t outputZeros = vdupq_n_f32(outputZero);
    for (; i + 16 <= count; i += 16) {
        const uint8x16_t bytes = vld1q_u8(src + i);
        const uint16x8_t low = vmovl_u8(vget_low_u8(bytes));
        const uint16x8_t high = vmovl_u8(vget_high_u8(bytes));
        const uint32x4_t ints[4] = {
            vmovl_u16(vget_low_u16(low)), vmovl_u16(vget_high_u16(low)),
            vmovl_u16(vget_low_u16(high)), vmovl_u16(vget_high_u16(high))};
        int32x4_t rounded[4];
        for (int j = 0; j < 4; ++j) {
            float32x4_t value = vsubq_f32(vcvtq_f32_u32(ints[j]), zeros);
            value = vmulq_n_f32(value, inputScale);
            value = vmulq_n_f32(value, gate);
            value = vaddq_f32(vmulq_n_f32(value, invOutputScale), outputZeros);
            value = vmaxq_f32(vdupq_n_f32(0.0f),
                              vminq_f32(vdupq_n_f32(255.0f), value));
            rounded[j] = vcvtnq_s32_f32(value);
        }
        const uint16x8_t low16 = vcombine_u16(vqmovun_s32(rounded[0]), vqmovun_s32(rounded[1]));
        const uint16x8_t high16 = vcombine_u16(vqmovun_s32(rounded[2]), vqmovun_s32(rounded[3]));
        vst1q_u8(dst + i, vcombine_u8(vqmovn_u16(low16), vqmovn_u16(high16)));
    }
#endif
    for (; i < count; ++i) {
        const float value = (((float)src[i] - inputZero) * inputScale * gate)
            * invOutputScale + outputZero;
        dst[i] = (uint8_t)std::max(0.0f, std::min(255.0f, std::nearbyint(value)));
    }
}

static void Eval(TFContext context, TFNode node) {
    const auto info = GetNodeInfo(node);
    const auto *param = (const Param *)GetNodeCustomParam(node);
    if (param == nullptr || param->channels < 1 || param->hidden < 1 || param->spatial < 1)
        throw std::runtime_error("SEBlock was not reshaped");
    const auto input = GetTensorByName(context, info.InputNames[0]);
    const auto output = GetTensorByName(context, info.OutputNames[0]);
    const int batches = param->batches;
    const int channels = param->channels;
    const int hiddenWidth = param->hidden;
    const int spatial = param->spatial;
    const int count = GetTensorCount(input, 0);
    if (count != batches * channels * spatial)
        throw std::runtime_error("SEBlock input shape changed after Reshape");

    const float *w1 = (const float *)GetTensordata(TensorOrParam(context, info.InputNames[1]));
    const float *b1 = (const float *)GetTensordata(TensorOrParam(context, info.InputNames[2]));
    const float *w2 = (const float *)GetTensordata(TensorOrParam(context, info.InputNames[3]));
    const float *b2 = (const float *)GetTensordata(TensorOrParam(context, info.InputNames[4]));
    if (w1 == nullptr || b1 == nullptr || w2 == nullptr || b2 == nullptr)
        throw std::runtime_error("SEBlock FC parameter data is unavailable");
#if defined(__aarch64__)
    auto *writable = (Param *)param;
    if (param->fp16Weights) {
        std::call_once(writable->packOnce, [&] {
            writable->w1Half.resize((size_t)hiddenWidth * channels);
            writable->w2Half.resize((size_t)channels * hiddenWidth);
            for (size_t i = 0; i < writable->w1Half.size(); ++i)
                writable->w1Half[i] = (__fp16)w1[i];
            for (size_t i = 0; i < writable->w2Half.size(); ++i)
                writable->w2Half[i] = (__fp16)w2[i];
        });
    }
#endif

    const float *source = nullptr;
    const uint8_t *byteSource = nullptr;
    std::vector<float> decoded;
    float inputScale = 0.0f, inputZero = 0.0f;
    if (GetTensorType(input) == TFCAPI_FLOAT) {
        source = (const float *)GetTensordata(input);
    } else if (GetTensorType(input) == TFCAPI_UINT8) {
        const auto quant = GetTensorQuantizeInfo(context, info.InputNames[0]);
        const auto &scales = GetQuantizationScale(quant);
        const auto &zeros = GetQuantizationZeroPoint(quant);
        byteSource = (const uint8_t *)GetTensordata(input);
        if (scales.size() == 1 && zeros.size() == 1) {
            inputScale = scales[0];
            inputZero = (float)zeros[0];
        } else {
            decoded.resize((size_t)count);
            DeQuantizeTensorData(decoded.data(), (uint8_t *)byteSource, count, quant);
            source = decoded.data();
            byteSource = nullptr;
        }
    } else {
        throw std::runtime_error("SEBlock unsupported input type");
    }
    if (source == nullptr && byteSource == nullptr)
        throw std::runtime_error("SEBlock input data is unavailable");

    std::vector<float> computed;
    float *destination = nullptr;
    uint8_t *quantDestination = nullptr;
    const auto outputType = GetTensorType(output);
    float outputScale = 0.0f, outputZero = 0.0f;
    bool directQuant = false;
    if (outputType == TFCAPI_UINT8 && byteSource != nullptr && param->fastQuant) {
        const auto quant = GetTensorQuantizeInfo(context, info.OutputNames[0]);
        const auto &scales = GetQuantizationScale(quant);
        const auto &zeros = GetQuantizationZeroPoint(quant);
        if (scales.size() == 1 && zeros.size() == 1 && scales[0] > 0) {
            outputScale = scales[0];
            outputZero = (float)zeros[0];
            directQuant = true;
        }
    }
    if (outputType == TFCAPI_FLOAT) {
        destination = (float *)GetTensordata(output);
    } else if (outputType == TFCAPI_UINT8) {
        if (directQuant) {
            quantDestination = (uint8_t *)GetTensordata(output);
        } else {
            computed.resize((size_t)count);
            destination = computed.data();
        }
    } else {
        throw std::runtime_error("SEBlock unsupported output type");
    }
    if (destination == nullptr && quantDestination == nullptr)
        throw std::runtime_error("SEBlock output data is unavailable");

    std::vector<float> pooled((size_t)channels);
    std::vector<float> hidden((size_t)hiddenWidth);
    std::vector<float> gate((size_t)channels);
    for (int batch = 0; batch < batches; ++batch) {
        const float *activation = source == nullptr ? nullptr
            : source + (size_t)batch * channels * spatial;
        const uint8_t *bytes = byteSource == nullptr ? nullptr
            : byteSource + (size_t)batch * channels * spatial;
        float *result = destination == nullptr ? nullptr
            : destination + (size_t)batch * channels * spatial;
        uint8_t *quantResult = quantDestination == nullptr ? nullptr
            : quantDestination + (size_t)batch * channels * spatial;
#if defined(_OPENMP)
#pragma omp parallel num_threads(param->threads) if(param->threads > 1 && channels * hiddenWidth >= 180000)
#endif
        {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int channel = 0; channel < channels; ++channel) {
            if (bytes != nullptr) {
                const uint64_t sum = SumBytes(bytes + (size_t)channel * spatial, spatial);
                pooled[channel] = ((float)sum / (float)spatial - inputZero) * inputScale;
            } else {
                pooled[channel] = SumFloat(activation + (size_t)channel * spatial, spatial)
                    / (float)spatial;
            }
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int index = 0; index < hiddenWidth; ++index) {
            float value = b1[index] +
#if defined(__aarch64__)
                (param->fp16Weights
                    ? DotHalf(pooled.data(), param->w1Half.data() + (size_t)index * channels, channels)
                    : Dot(pooled.data(), w1 + (size_t)index * channels, channels));
#else
                Dot(pooled.data(), w1 + (size_t)index * channels, channels);
#endif
            hidden[index] = std::fmax(value, 0.0f);
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int channel = 0; channel < channels; ++channel) {
            const float value = b2[channel] +
#if defined(__aarch64__)
                (param->fp16Weights
                    ? DotHalf(hidden.data(), param->w2Half.data() + (size_t)channel * hiddenWidth, hiddenWidth)
                    : Dot(hidden.data(), w2 + (size_t)channel * hiddenWidth, hiddenWidth));
#else
                Dot(hidden.data(), w2 + (size_t)channel * hiddenWidth, hiddenWidth);
#endif
            gate[channel] = 1.0f / (1.0f + std::exp(-value));
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int channel = 0; channel < channels; ++channel) {
            if (quantResult != nullptr) {
                MultiplyBytesToQuant(quantResult + (size_t)channel * spatial,
                                     bytes + (size_t)channel * spatial,
                                     inputScale, inputZero, gate[channel],
                                     outputScale, outputZero, spatial);
            } else if (bytes != nullptr) {
                MultiplyBytesToFloat(result + (size_t)channel * spatial,
                                     bytes + (size_t)channel * spatial,
                                     inputScale, inputZero, gate[channel], spatial);
            } else {
                MultiplyFloat(result + (size_t)channel * spatial,
                              activation + (size_t)channel * spatial,
                              gate[channel], spatial);
            }
        }
        }
    }
    if (outputType == TFCAPI_UINT8 && !directQuant) {
        const auto quant = GetTensorQuantizeInfo(context, info.OutputNames[0]);
        QuantizeTensorData((uint8_t *)GetTensordata(output), computed.data(), count, quant);
    }
}

static void Free(TFContext, TFNode node) {
    FreeNodeCustomParam(node, [](void *value) { delete (Param *)value; });
}

}  // namespace SEBlock

RegistOp(SEBlock).Set(SEBlock::Prepare, SEBlock::Reshape,
                       SEBlock::Eval, SEBlock::Free);
}  // namespace TFDLOP

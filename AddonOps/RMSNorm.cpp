// High-throughput RMSNorm custom operator.
//
// Input/output: arbitrary rank, with normalization over a configurable axis.
// FP32 and native FP16 tensors use AArch64 NEON and OpenMP parallelism.
// The axis=1 [B,C,S] path vectorizes across adjacent tokens while reducing C,
// so a Conv-native residual stream does not need a physical Transpose.
// UINT8 keeps a direct scalar-qinfo path and a generic compatibility fallback.

#ifndef NPU40T_RMSNORM_H
#define NPU40T_RMSNORM_H

#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

using namespace TFDL_CAPI;

namespace TFDLOP {
namespace RMSNorm {

struct Param {
    float eps = 1e-5f;
    int threads = 0;
    int axis = -1;
};

static constexpr long long kMinParallelElements = 32 * 1024;

static void Prepare(TFContext tfContext, TFNode node) {
    (void)tfContext;
    string error;
    const json11::Json config = json11::Json::parse(
        GetNodeCustomJsonStr(node), error);
    if (!error.empty()) {
        throw std::runtime_error("RMSNorm invalid JSON: " + error);
    }
    Param *param = new Param();
    if (!config["eps"].is_null()) {
        param->eps = (float)config["eps"].number_value();
    }
    param->threads = config["threads"].int_value();
    if (!config["axis"].is_null()) {
        param->axis = config["axis"].int_value();
    }
    if (!(param->eps > 0.f) || !std::isfinite(param->eps)) {
        delete param;
        throw std::runtime_error("RMSNorm eps must be finite and positive");
    }
    if (param->threads < 0) {
        delete param;
        throw std::runtime_error("RMSNorm threads must be non-negative");
    }
    FreeNodeCustomParam(node, [](void *value) { delete (Param *)value; });
    NewNodeCustomParam(node, [&param]() -> void * { return param; });
}

struct AxisLayout {
    int axis;
    int outer;
    int width;
    int inner;
    int count;
};

static AxisLayout ResolveAxisLayout(
        const vector<int> &shape, int requestedAxis) {
    if (shape.empty()) {
        throw std::runtime_error("RMSNorm requires a non-empty input shape");
    }
    const int rank = (int)shape.size();
    const int axis = requestedAxis < 0 ? requestedAxis + rank : requestedAxis;
    if (axis < 0 || axis >= rank) {
        throw std::runtime_error("RMSNorm axis is outside the input rank");
    }
    long long outer = 1;
    long long inner = 1;
    for (int index = 0; index < rank; ++index) {
        if (shape[index] <= 0) {
            throw std::runtime_error(
                "RMSNorm requires strictly positive dimensions");
        }
        if (index < axis) outer *= shape[index];
        if (index > axis) inner *= shape[index];
    }
    const long long width = shape[axis];
    const long long count = outer * width * inner;
    if (outer > INT_MAX || width > INT_MAX || inner > INT_MAX
            || count <= 0 || count > INT_MAX) {
        throw std::runtime_error("RMSNorm tensor is too large");
    }
    return {
        axis,
        (int)outer,
        (int)width,
        (int)inner,
        (int)count,
    };
}

static void Reshape(TFContext tfContext, TFNode node) {
    const auto info = GetNodeInfo(node);
    TFCHECK_GE(info.InputNames.size(), 1);
    TFCHECK_LE(info.InputNames.size(), 2);
    TFCHECK_EQ(info.OutputNames.size(), 1);
    auto input = GetTensorByName(tfContext, info.InputNames[0]);
    auto output = GetTensorByName(tfContext, info.OutputNames[0]);
    const auto shape = GetTensorShape(input);
    const Param *param = (const Param *)GetNodeCustomParam(node);
    if (param == nullptr) {
        throw std::runtime_error("RMSNorm parameter state is missing");
    }
    const AxisLayout layout = ResolveAxisLayout(shape, param->axis);
    const auto dtype = GetTensorType(input);
    if (dtype != TFCAPI_FLOAT && dtype != TFCAPI_FLOAT16
            && dtype != TFCAPI_UINT8) {
        throw std::runtime_error(
            "RMSNorm supports only FP32, FP16, and UINT8 tensors");
    }
    if (info.InputNames.size() == 2) {
        auto weight = GetTensorByName(tfContext, info.InputNames[1]);
        if (!weight.IsValid()) weight = GetParam(tfContext, info.InputNames[1]);
        if (!weight.IsValid() || GetTensorType(weight) != dtype
                || GetTensorCount(weight, 0) != layout.width) {
            throw std::runtime_error(
                "RMSNorm affine weight must match the normalized axis");
        }
        if (dtype == TFCAPI_UINT8) {
            throw std::runtime_error(
                "RMSNorm affine weight is only supported for FP32/FP16");
        }
    }
    ReSizeTensor(output, shape);
    SetTensorType(output, dtype);
}

template <typename Function>
static inline void ForEachRow(
        int rows, long long elements, int requestedThreads,
        const Function &function) {
#ifdef _OPENMP
    const int threads = requestedThreads > 0
        ? requestedThreads : omp_get_max_threads();
    if (threads > 1 && elements >= kMinParallelElements
            && !omp_in_parallel()) {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int row = 0; row < rows; ++row) function(row);
        return;
    }
#else
    (void)elements;
    (void)requestedThreads;
#endif
    for (int row = 0; row < rows; ++row) function(row);
}

static inline float SumSquaresFloat(const float *input, int count) {
    int index = 0;
#ifdef __aarch64__
    float32x4_t sum0 = vdupq_n_f32(0.f);
    float32x4_t sum1 = vdupq_n_f32(0.f);
    float32x4_t sum2 = vdupq_n_f32(0.f);
    float32x4_t sum3 = vdupq_n_f32(0.f);
    for (; index + 16 <= count; index += 16) {
        const float32x4_t value0 = vld1q_f32(input + index);
        const float32x4_t value1 = vld1q_f32(input + index + 4);
        const float32x4_t value2 = vld1q_f32(input + index + 8);
        const float32x4_t value3 = vld1q_f32(input + index + 12);
        sum0 = vfmaq_f32(sum0, value0, value0);
        sum1 = vfmaq_f32(sum1, value1, value1);
        sum2 = vfmaq_f32(sum2, value2, value2);
        sum3 = vfmaq_f32(sum3, value3, value3);
    }
    float sum = vaddvq_f32(vaddq_f32(
        vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
#else
    float sum = 0.f;
#endif
    for (; index < count; ++index) sum += input[index] * input[index];
    return sum;
}

static inline void ScaleFloat(
        const float *input, float *output, int count, float scale) {
    int index = 0;
#ifdef __aarch64__
    const float32x4_t factor = vdupq_n_f32(scale);
    for (; index + 16 <= count; index += 16) {
        vst1q_f32(output + index,
                  vmulq_f32(vld1q_f32(input + index), factor));
        vst1q_f32(output + index + 4,
                  vmulq_f32(vld1q_f32(input + index + 4), factor));
        vst1q_f32(output + index + 8,
                  vmulq_f32(vld1q_f32(input + index + 8), factor));
        vst1q_f32(output + index + 12,
                  vmulq_f32(vld1q_f32(input + index + 12), factor));
    }
#endif
    for (; index < count; ++index) output[index] = input[index] * scale;
}

static inline void MultiplyFloatWeights(
        float *values, const float *weight, int count) {
    int index = 0;
#ifdef __aarch64__
    for (; index + 16 <= count; index += 16) {
        vst1q_f32(values + index,
                  vmulq_f32(vld1q_f32(values + index),
                            vld1q_f32(weight + index)));
        vst1q_f32(values + index + 4,
                  vmulq_f32(vld1q_f32(values + index + 4),
                            vld1q_f32(weight + index + 4)));
        vst1q_f32(values + index + 8,
                  vmulq_f32(vld1q_f32(values + index + 8),
                            vld1q_f32(weight + index + 8)));
        vst1q_f32(values + index + 12,
                  vmulq_f32(vld1q_f32(values + index + 12),
                            vld1q_f32(weight + index + 12)));
    }
#endif
    for (; index < count; ++index) values[index] *= weight[index];
}

static inline void NormalizeFloatRow(
        const float *input, float *output, int width, float eps) {
    const float sum = SumSquaresFloat(input, width);
    const float inverse = 1.f / std::sqrt(sum / (float)width + eps);
    ScaleFloat(input, output, width, inverse);
}

static void NormalizeFloat(
        const float *input, float *output, int rows, int width,
        float eps, int threads) {
    ForEachRow(rows, (long long)rows * width, threads, [=](int row) {
        NormalizeFloatRow(
            input + (long long)row * width,
            output + (long long)row * width,
            width, eps);
    });
}

static void NormalizeFloatAxis(
        const float *input, float *output,
        int outer, int width, int inner, float eps, int threads) {
    if (inner == 1) {
        NormalizeFloat(input, output, outer, width, eps, threads);
        return;
    }
#ifdef __aarch64__
    const int vectorWidth = 4;
    const int blocks = (inner + vectorWidth - 1) / vectorWidth;
    ForEachRow(
        outer * blocks,
        (long long)outer * width * inner,
        threads,
        [=](int task) {
            const int outerIndex = task / blocks;
            const int begin = (task % blocks) * vectorWidth;
            const int valid = std::min(vectorWidth, inner - begin);
            const long long base = (long long)outerIndex * width * inner + begin;
            if (valid == vectorWidth) {
                // Match NormalizeFloatRow's legacy 16-channel reduction
                // order. Fixed INT8 ranges can amplify even one-ULP RMSNorm
                // changes at outlier layers, so a single sequential channel
                // accumulator is not numerically interchangeable here.
                float32x4_t sums[16];
                for (int slot = 0; slot < 16; ++slot) {
                    sums[slot] = vdupq_n_f32(0.f);
                }
                int channel = 0;
                for (; channel + 16 <= width; channel += 16) {
                    for (int slot = 0; slot < 16; ++slot) {
                        const float32x4_t value = vld1q_f32(
                            input + base
                            + (long long)(channel + slot) * inner);
                        sums[slot] = vfmaq_f32(
                            sums[slot], value, value);
                    }
                }
                const float32x4_t lane0 = vaddq_f32(
                    vaddq_f32(sums[0], sums[4]),
                    vaddq_f32(sums[8], sums[12]));
                const float32x4_t lane1 = vaddq_f32(
                    vaddq_f32(sums[1], sums[5]),
                    vaddq_f32(sums[9], sums[13]));
                const float32x4_t lane2 = vaddq_f32(
                    vaddq_f32(sums[2], sums[6]),
                    vaddq_f32(sums[10], sums[14]));
                const float32x4_t lane3 = vaddq_f32(
                    vaddq_f32(sums[3], sums[7]),
                    vaddq_f32(sums[11], sums[15]));
                float32x4_t sum = vaddq_f32(
                    vaddq_f32(lane0, lane1),
                    vaddq_f32(lane2, lane3));
                for (; channel < width; ++channel) {
                    const float32x4_t value = vld1q_f32(
                        input + base + (long long)channel * inner);
                    sum = vfmaq_f32(sum, value, value);
                }
                alignas(16) float sumValues[4];
                alignas(16) float factors[4];
                vst1q_f32(sumValues, sum);
                for (int lane = 0; lane < 4; ++lane) {
                    factors[lane] = 1.f / std::sqrt(
                        sumValues[lane] / (float)width + eps);
                }
                const float32x4_t factor = vld1q_f32(factors);
                for (int channel = 0; channel < width; ++channel) {
                    const long long offset = base + (long long)channel * inner;
                    vst1q_f32(
                        output + offset,
                        vmulq_f32(vld1q_f32(input + offset), factor));
                }
                return;
            }
            for (int lane = 0; lane < valid; ++lane) {
                float sum = 0.f;
                const long long tokenBase = base + lane;
                for (int channel = 0; channel < width; ++channel) {
                    const float value = input[
                        tokenBase + (long long)channel * inner];
                    sum += value * value;
                }
                const float inverse = 1.f / std::sqrt(
                    sum / (float)width + eps);
                for (int channel = 0; channel < width; ++channel) {
                    const long long offset =
                        tokenBase + (long long)channel * inner;
                    output[offset] = input[offset] * inverse;
                }
            }
        });
#else
    ForEachRow(
        outer * inner,
        (long long)outer * width * inner,
        threads,
        [=](int row) {
            const int outerIndex = row / inner;
            const int innerIndex = row % inner;
            const long long base =
                (long long)outerIndex * width * inner + innerIndex;
            float sum = 0.f;
            for (int channel = 0; channel < width; ++channel) {
                const float value = input[base + (long long)channel * inner];
                sum += value * value;
            }
            const float inverse = 1.f / std::sqrt(
                sum / (float)width + eps);
            for (int channel = 0; channel < width; ++channel) {
                const long long offset = base + (long long)channel * inner;
                output[offset] = input[offset] * inverse;
            }
        });
#endif
}

static void ApplyFloatAffine(
        float *output, const float *weight,
        int outer, int width, int inner, int threads) {
    if (inner == 1) {
        // Last-axis RMSNorm stores every learned scale contiguously. Work by
        // normalized row instead of launching one one-element task per
        // channel; this is the Q/K attention layout [B,H,S,D].
        ForEachRow(
            outer,
            (long long)outer * width,
            threads,
            [=](int row) {
                MultiplyFloatWeights(
                    output + (long long)row * width, weight, width);
            });
        return;
    }
    ForEachRow(
        outer * width,
        (long long)outer * width * inner,
        threads,
        [=](int row) {
            const int channel = row % width;
            float *values = output + (long long)row * inner;
            ScaleFloat(values, values, inner, weight[channel]);
        });
}

static float Fp16ToFloat(uint16_t half) {
    uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            bits = sign | (uint32_t)(127 - 15 - shift) << 23
                | mantissa << 13;
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + 112u) << 23 | mantissa << 13;
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t FloatToFp16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    int exponent = (int)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x007fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        mantissa = (mantissa | 0x00800000u) >> (1 - exponent);
        return (uint16_t)(sign + ((mantissa + 0x00001000u) >> 13));
    }
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    mantissa += 0x00001000u;
    if (mantissa & 0x00800000u) {
        mantissa = 0;
        if (++exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(
        sign | (uint16_t)(exponent << 10) | (uint16_t)(mantissa >> 13));
}

static inline void NormalizeFp16Row(
        const uint16_t *inputStorage, uint16_t *outputStorage,
        int width, float eps) {
    int index = 0;
    float sum = 0.f;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const __fp16 *input = reinterpret_cast<const __fp16 *>(inputStorage);
    __fp16 *output = reinterpret_cast<__fp16 *>(outputStorage);
    float32x4_t sum0 = vdupq_n_f32(0.f);
    float32x4_t sum1 = vdupq_n_f32(0.f);
    float32x4_t sum2 = vdupq_n_f32(0.f);
    float32x4_t sum3 = vdupq_n_f32(0.f);
    for (; index + 16 <= width; index += 16) {
        const float16x8_t half0 = vld1q_f16(input + index);
        const float16x8_t half1 = vld1q_f16(input + index + 8);
        const float32x4_t value0 = vcvt_f32_f16(vget_low_f16(half0));
        const float32x4_t value1 = vcvt_f32_f16(vget_high_f16(half0));
        const float32x4_t value2 = vcvt_f32_f16(vget_low_f16(half1));
        const float32x4_t value3 = vcvt_f32_f16(vget_high_f16(half1));
        sum0 = vfmaq_f32(sum0, value0, value0);
        sum1 = vfmaq_f32(sum1, value1, value1);
        sum2 = vfmaq_f32(sum2, value2, value2);
        sum3 = vfmaq_f32(sum3, value3, value3);
    }
    sum = vaddvq_f32(vaddq_f32(
        vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
    for (int tail = index; tail < width; ++tail) {
        const float value = (float)input[tail];
        sum += value * value;
    }
    const float inverse = 1.f / std::sqrt(sum / (float)width + eps);
    const float32x4_t factor = vdupq_n_f32(inverse);
    index = 0;
    for (; index + 8 <= width; index += 8) {
        const float16x8_t half = vld1q_f16(input + index);
        const float32x4_t low = vmulq_f32(
            vcvt_f32_f16(vget_low_f16(half)), factor);
        const float32x4_t high = vmulq_f32(
            vcvt_f32_f16(vget_high_f16(half)), factor);
        vst1q_f16(output + index, vcombine_f16(
            vcvt_f16_f32(low), vcvt_f16_f32(high)));
    }
    for (; index < width; ++index) output[index] = (__fp16)(input[index] * inverse);
#else
    for (; index < width; ++index) {
        const float value = Fp16ToFloat(inputStorage[index]);
        sum += value * value;
    }
    const float inverse = 1.f / std::sqrt(sum / (float)width + eps);
    for (index = 0; index < width; ++index) {
        outputStorage[index] = FloatToFp16(
            Fp16ToFloat(inputStorage[index]) * inverse);
    }
#endif
}

static void NormalizeFp16(
        const uint16_t *input, uint16_t *output, int rows, int width,
        float eps, int threads) {
    ForEachRow(rows, (long long)rows * width, threads, [=](int row) {
        NormalizeFp16Row(
            input + (long long)row * width,
            output + (long long)row * width,
            width, eps);
    });
}

static void NormalizeFp16Axis(
        const uint16_t *inputStorage, uint16_t *outputStorage,
        int outer, int width, int inner, float eps, int threads) {
    if (inner == 1) {
        NormalizeFp16(
            inputStorage, outputStorage, outer, width, eps, threads);
        return;
    }
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const __fp16 *input = reinterpret_cast<const __fp16 *>(inputStorage);
    __fp16 *output = reinterpret_cast<__fp16 *>(outputStorage);
    // Four tokens leave room for the 16 accumulators needed to reproduce the
    // contiguous last-axis kernel's reduction tree without register spills.
    const int vectorWidth = 4;
    const int blocks = (inner + vectorWidth - 1) / vectorWidth;
    ForEachRow(
        outer * blocks,
        (long long)outer * width * inner,
        threads,
        [=](int task) {
            const int outerIndex = task / blocks;
            const int begin = (task % blocks) * vectorWidth;
            const int valid = std::min(vectorWidth, inner - begin);
            const long long base = (long long)outerIndex * width * inner + begin;
            if (valid == vectorWidth) {
                float32x4_t sums[16];
                for (int slot = 0; slot < 16; ++slot) {
                    sums[slot] = vdupq_n_f32(0.f);
                }
                int channel = 0;
                for (; channel + 16 <= width; channel += 16) {
                    for (int slot = 0; slot < 16; ++slot) {
                        const float32x4_t value = vcvt_f32_f16(vld1_f16(
                            input + base
                            + (long long)(channel + slot) * inner));
                        sums[slot] = vfmaq_f32(
                            sums[slot], value, value);
                    }
                }
                const float32x4_t lane0 = vaddq_f32(
                    vaddq_f32(sums[0], sums[4]),
                    vaddq_f32(sums[8], sums[12]));
                const float32x4_t lane1 = vaddq_f32(
                    vaddq_f32(sums[1], sums[5]),
                    vaddq_f32(sums[9], sums[13]));
                const float32x4_t lane2 = vaddq_f32(
                    vaddq_f32(sums[2], sums[6]),
                    vaddq_f32(sums[10], sums[14]));
                const float32x4_t lane3 = vaddq_f32(
                    vaddq_f32(sums[3], sums[7]),
                    vaddq_f32(sums[11], sums[15]));
                float32x4_t sum = vaddq_f32(
                    vaddq_f32(lane0, lane1),
                    vaddq_f32(lane2, lane3));
                for (; channel < width; ++channel) {
                    const float32x4_t value = vcvt_f32_f16(vld1_f16(
                        input + base + (long long)channel * inner));
                    sum = vfmaq_f32(sum, value, value);
                }
                alignas(16) float sumValues[4];
                alignas(16) float factors[4];
                vst1q_f32(sumValues, sum);
                for (int lane = 0; lane < 4; ++lane) {
                    factors[lane] = 1.f / std::sqrt(
                        sumValues[lane] / (float)width + eps);
                }
                const float32x4_t factor = vld1q_f32(factors);
                for (int channel = 0; channel < width; ++channel) {
                    const long long offset = base + (long long)channel * inner;
                    const float32x4_t normalized = vmulq_f32(
                        vcvt_f32_f16(vld1_f16(input + offset)), factor);
                    vst1_f16(output + offset, vcvt_f16_f32(normalized));
                }
                return;
            }
            for (int lane = 0; lane < valid; ++lane) {
                float sum = 0.f;
                const long long tokenBase = base + lane;
                for (int channel = 0; channel < width; ++channel) {
                    const float value = (float)input[
                        tokenBase + (long long)channel * inner];
                    sum += value * value;
                }
                const float inverse = 1.f / std::sqrt(
                    sum / (float)width + eps);
                for (int channel = 0; channel < width; ++channel) {
                    const long long offset =
                        tokenBase + (long long)channel * inner;
                    output[offset] = (__fp16)((float)input[offset] * inverse);
                }
            }
        });
#else
    ForEachRow(
        outer * inner,
        (long long)outer * width * inner,
        threads,
        [=](int row) {
            const int outerIndex = row / inner;
            const int innerIndex = row % inner;
            const long long base =
                (long long)outerIndex * width * inner + innerIndex;
            float sum = 0.f;
            for (int channel = 0; channel < width; ++channel) {
                const float value = Fp16ToFloat(
                    inputStorage[base + (long long)channel * inner]);
                sum += value * value;
            }
            const float inverse = 1.f / std::sqrt(
                sum / (float)width + eps);
            for (int channel = 0; channel < width; ++channel) {
                const long long offset = base + (long long)channel * inner;
                outputStorage[offset] = FloatToFp16(
                    Fp16ToFloat(inputStorage[offset]) * inverse);
            }
        });
#endif
}

static inline void MultiplyFp16Weights(
        uint16_t *valuesStorage, const uint16_t *weightStorage, int count) {
    int index = 0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    __fp16 *values = reinterpret_cast<__fp16 *>(valuesStorage);
    const __fp16 *weight = reinterpret_cast<const __fp16 *>(weightStorage);
    for (; index + 16 <= count; index += 16) {
        vst1q_f16(values + index,
                  vmulq_f16(vld1q_f16(values + index),
                            vld1q_f16(weight + index)));
        vst1q_f16(values + index + 8,
                  vmulq_f16(vld1q_f16(values + index + 8),
                            vld1q_f16(weight + index + 8)));
    }
    for (; index < count; ++index) values[index] *= weight[index];
#else
    for (; index < count; ++index) {
        valuesStorage[index] = FloatToFp16(
            Fp16ToFloat(valuesStorage[index])
            * Fp16ToFloat(weightStorage[index]));
    }
#endif
}

static void ApplyFp16Affine(
        uint16_t *outputStorage, const uint16_t *weightStorage,
        int outer, int width, int inner, int threads) {
    if (inner == 1) {
        ForEachRow(
            outer,
            (long long)outer * width,
            threads,
            [=](int row) {
                MultiplyFp16Weights(
                    outputStorage + (long long)row * width,
                    weightStorage,
                    width);
            });
        return;
    }
    ForEachRow(
        outer * width,
        (long long)outer * width * inner,
        threads,
        [=](int row) {
            const int channel = row % width;
            uint16_t *valuesStorage = outputStorage + (long long)row * inner;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            __fp16 *values = reinterpret_cast<__fp16 *>(valuesStorage);
            const __fp16 factor = reinterpret_cast<const __fp16 *>(
                weightStorage)[channel];
            const float16x8_t vectorFactor = vdupq_n_f16(factor);
            int index = 0;
            for (; index + 8 <= inner; index += 8) {
                vst1q_f16(
                    values + index,
                    vmulq_f16(vld1q_f16(values + index), vectorFactor));
            }
            for (; index < inner; ++index) values[index] *= factor;
#else
            const float factor = Fp16ToFloat(weightStorage[channel]);
            for (int index = 0; index < inner; ++index) {
                valuesStorage[index] = FloatToFp16(
                    Fp16ToFloat(valuesStorage[index]) * factor);
            }
#endif
        });
}

static void NormalizeUint8Fallback(
        uint8_t *input, uint8_t *output, int count,
        int outer, int width, int inner, float eps, int threads,
        Quantization inputQuant, Quantization outputQuant) {
    std::vector<float> decoded((size_t)count);
    std::vector<float> normalized((size_t)count);
    DeQuantizeTensorData(decoded.data(), input, count, inputQuant);
    NormalizeFloatAxis(
        decoded.data(), normalized.data(),
        outer, width, inner, eps, threads);
    QuantizeTensorData(output, normalized.data(), count, outputQuant);
}

static void Eval(TFContext tfContext, TFNode node) {
    const Param *param = (const Param *)GetNodeCustomParam(node);
    if (param == nullptr) {
        throw std::runtime_error("RMSNorm parameter state is missing");
    }
    const auto info = GetNodeInfo(node);
    auto input = GetTensorByName(tfContext, info.InputNames[0]);
    auto output = GetTensorByName(tfContext, info.OutputNames[0]);
    const auto shape = GetTensorShape(input);
    const AxisLayout layout = ResolveAxisLayout(shape, param->axis);
    const long countLong = GetTensorCount(input, 0);
    if (countLong != layout.count) {
        throw std::runtime_error("RMSNorm got an invalid tensor size");
    }
    const auto dtype = GetTensorType(input);
    if (dtype != GetTensorType(output)) {
        throw std::runtime_error("RMSNorm input/output dtype mismatch");
    }
    if (dtype == TFCAPI_FLOAT) {
        float *outputData = (float *)GetTensordata(output);
        NormalizeFloatAxis(
            (const float *)GetTensordata(input),
            outputData,
            layout.outer, layout.width, layout.inner,
            param->eps, param->threads);
        if (info.InputNames.size() == 2) {
            auto weight = GetTensorByName(tfContext, info.InputNames[1]);
            if (!weight.IsValid()) weight = GetParam(tfContext, info.InputNames[1]);
            ApplyFloatAffine(
                outputData, (const float *)GetTensordata(weight),
                layout.outer, layout.width, layout.inner, param->threads);
        }
        return;
    }
    if (dtype == TFCAPI_FLOAT16) {
        uint16_t *outputData = (uint16_t *)GetTensordata(output);
        NormalizeFp16Axis(
            (const uint16_t *)GetTensordata(input),
            outputData,
            layout.outer, layout.width, layout.inner,
            param->eps, param->threads);
        if (info.InputNames.size() == 2) {
            auto weight = GetTensorByName(tfContext, info.InputNames[1]);
            if (!weight.IsValid()) weight = GetParam(tfContext, info.InputNames[1]);
            ApplyFp16Affine(
                outputData, (const uint16_t *)GetTensordata(weight),
                layout.outer, layout.width, layout.inner, param->threads);
        }
        return;
    }
    if (dtype == TFCAPI_UINT8) {
        auto inputQuant = GetTensorQuantizeInfo(
            tfContext, info.InputNames[0]);
        auto outputQuant = GetTensorQuantizeInfo(
            tfContext, info.OutputNames[0]);
        if (!inputQuant.IsValid() || !outputQuant.IsValid()) {
            throw std::runtime_error(
                "RMSNorm UINT8 requires input/output quantization info");
        }
        NormalizeUint8Fallback(
            (uint8_t *)GetTensordata(input),
            (uint8_t *)GetTensordata(output),
            layout.count,
            layout.outer, layout.width, layout.inner,
            param->eps, param->threads,
            inputQuant, outputQuant);
        return;
    }
    throw std::runtime_error("RMSNorm encountered an unsupported dtype");
}

static void Free(TFContext tfContext, TFNode node) {
    (void)tfContext;
    FreeNodeCustomParam(node, [](void *value) { delete (Param *)value; });
}

}  // namespace RMSNorm

RegistOp(RMSNorm)
.Set(RMSNorm::Prepare, RMSNorm::Reshape, RMSNorm::Eval, RMSNorm::Free);

}  // namespace TFDLOP

#endif  // NPU40T_RMSNORM_H

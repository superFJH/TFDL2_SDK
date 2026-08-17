// ARM-optimized causal Softmax for UINT8 attention scores.
//
// Input/output: [..., S, S] UINT8.  The first input accepts either scalar or
// one-qinfo-per-outer-row metadata directly.  A second optional FP32 row-scale
// input remains supported only so FBs produced for older SDK schedulers keep
// working.  Softmax is invariant to each row's zero point.  The causal rule
// includes the diagonal: row q consumes keys [0, q].
//
// The implementation never materializes a floating score matrix.  For each
// row it finds the maximum UINT8 code, evaluates exp(-scale) once, builds the
// remaining code-span exponent values by recurrence, and directly emits
// quantized probabilities. Rows are independent and therefore parallelized
// with OpenMP; AArch64 uses NEON for the row minimum/maximum scan.

#ifndef NPU40T_ARM_CAUSAL_MASK_SOFTMAX_H
#define NPU40T_ARM_CAUSAL_MASK_SOFTMAX_H

#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

using namespace TFDL_CAPI;

namespace TFDLOP {
namespace ArmCausalMaskSoftmax {

struct Param {
    int threads = 0;
};

static void Prepare(TFContext tfContext, TFNode node) {
    json11::Json config;
    string error;
    config = json11::Json::parse(GetNodeCustomJsonStr(node), error);
    if (!error.empty()) {
        throw std::runtime_error(
            "ArmCausalMaskSoftmax invalid JSON: " + error);
    }
    Param *param = new Param();
    param->threads = config["threads"].int_value();
    if (param->threads < 0) {
        delete param;
        throw std::runtime_error(
            "ArmCausalMaskSoftmax threads must be non-negative");
    }
    FreeNodeCustomParam(node, [](void *value) { delete (Param *)value; });
    NewNodeCustomParam(node, [&param]() -> void * { return param; });
}

static int ValidateShape(const vector<int> &shape) {
    if (shape.size() != 3 && shape.size() != 4) {
        throw std::runtime_error(
            "ArmCausalMaskSoftmax expects [H,S,S] or [B,H,S,S]");
    }
    const int queryRows = shape[shape.size() - 2];
    const int key = shape[shape.size() - 1];
    const int sequence = key;
    if (sequence <= 0 || key != sequence || queryRows <= 0
            || queryRows % sequence != 0) {
        throw std::runtime_error(
            "ArmCausalMaskSoftmax requires [..., S, S] or grouped "
            "[..., R*S, S] trailing axes");
    }
    return sequence;
}

static void Reshape(TFContext tfContext, TFNode node) {
    const auto info = GetNodeInfo(node);
    TFCHECK_GE(info.InputNames.size(), 1);
    TFCHECK_LE(info.InputNames.size(), 2);
    TFCHECK_EQ(info.OutputNames.size(), 1);
    auto input = GetTensorByName(tfContext, info.InputNames[0]);
    auto output = GetTensorByName(tfContext, info.OutputNames[0]);
    if (GetTensorType(input) != TFCAPI_UINT8) {
        throw std::runtime_error(
            "ArmCausalMaskSoftmax only supports UINT8 input");
    }
    const auto shape = GetTensorShape(input);
    ValidateShape(shape);
    if (info.InputNames.size() == 2) {
        auto scale = GetTensorByName(tfContext, info.InputNames[1]);
        if (!scale.IsValid()) scale = GetParam(tfContext, info.InputNames[1]);
        if (!scale.IsValid() || GetTensorType(scale) != TFCAPI_FLOAT) {
            throw std::runtime_error(
                "ArmCausalMaskSoftmax row-scale sidecar must be FP32");
        }
        const long rows = GetTensorCount(input, 0) / shape.back();
        if (GetTensorCount(scale, 0) != rows) {
            throw std::runtime_error(
                "ArmCausalMaskSoftmax row-scale sidecar length mismatch");
        }
    }
    ReSizeTensor(output, shape);
    SetTensorType(output, TFCAPI_UINT8);
}

struct CodeRange {
    uint8_t minimum;
    uint8_t maximum;
};

static inline CodeRange RowCodeRange(const uint8_t *input, int count) {
    int index = 0;
    uint8_t minimum = 255;
    uint8_t maximum = 0;
#ifdef __aarch64__
    uint8x16_t vectorMinimum = vdupq_n_u8(255);
    uint8x16_t vectorMaximum = vdupq_n_u8(0);
    for (; index + 16 <= count; index += 16) {
        const uint8x16_t value = vld1q_u8(input + index);
        vectorMinimum = vminq_u8(vectorMinimum, value);
        vectorMaximum = vmaxq_u8(vectorMaximum, value);
    }
    minimum = vminvq_u8(vectorMinimum);
    maximum = vmaxvq_u8(vectorMaximum);
#endif
    for (; index < count; ++index) {
        minimum = std::min(minimum, input[index]);
        maximum = std::max(maximum, input[index]);
    }
    return {minimum, maximum};
}

static inline uint8_t QuantizeProbability(
        float probability, float inverseOutputScale, int outputZero) {
    float encoded = std::nearbyint(
        probability * inverseOutputScale + (float)outputZero);
    if (encoded <= 0.f) return 0;
    if (encoded >= 255.f) return 255;
    return (uint8_t)encoded;
}

static inline void ProcessRow(
        const uint8_t *input, uint8_t *output,
        int sequence, int query,
        float exponentRatio, float inverseOutputScale, int outputZero) {
    const int valid = query + 1;
    const CodeRange range = RowCodeRange(input, valid);
    const int minimum = (int)range.minimum;
    const int maximum = (int)range.maximum;

    // Softmax is invariant to the input zero point.  Since affine UINT8
    // scales are positive, subtracting the maximum real score is exactly
    // (code - maximum) * inputScale.
    alignas(64) float exponent[256];
    exponent[maximum] = 1.f;
    for (int code = maximum - 1; code >= minimum; --code) {
        exponent[code] = exponent[code + 1] * exponentRatio;
    }

    // Eight accumulators hide dependent-add and random-LUT-load latency on
    // Cortex-A77-class cores.
    float sum0 = 0.f;
    float sum1 = 0.f;
    float sum2 = 0.f;
    float sum3 = 0.f;
    float sum4 = 0.f;
    float sum5 = 0.f;
    float sum6 = 0.f;
    float sum7 = 0.f;
    int key = 0;
    for (; key + 8 <= valid; key += 8) {
        sum0 += exponent[input[key]];
        sum1 += exponent[input[key + 1]];
        sum2 += exponent[input[key + 2]];
        sum3 += exponent[input[key + 3]];
        sum4 += exponent[input[key + 4]];
        sum5 += exponent[input[key + 5]];
        sum6 += exponent[input[key + 6]];
        sum7 += exponent[input[key + 7]];
    }
    float sum = ((sum0 + sum1) + (sum2 + sum3))
        + ((sum4 + sum5) + (sum6 + sum7));
    for (; key < valid; ++key) sum += exponent[input[key]];
    // exponent[maximum] is exactly one, so validated positive scales make an
    // invalid sum unreachable without memory corruption.
    const float quantizeFactor = inverseOutputScale / sum;
    const int codeSpan = maximum - minimum + 1;
    if (codeSpan <= valid) {
        // Quantize each possible code once, then the output pass becomes a
        // byte LUT lookup. This removes hundreds of float round/conversions
        // from the long causal rows that dominate prefill.
        alignas(64) uint8_t encoded[256];
        for (int code = minimum; code <= maximum; ++code) {
            encoded[code] = QuantizeProbability(
                exponent[code], quantizeFactor, outputZero);
        }
        key = 0;
        for (; key + 16 <= valid; key += 16) {
            output[key] = encoded[input[key]];
            output[key + 1] = encoded[input[key + 1]];
            output[key + 2] = encoded[input[key + 2]];
            output[key + 3] = encoded[input[key + 3]];
            output[key + 4] = encoded[input[key + 4]];
            output[key + 5] = encoded[input[key + 5]];
            output[key + 6] = encoded[input[key + 6]];
            output[key + 7] = encoded[input[key + 7]];
            output[key + 8] = encoded[input[key + 8]];
            output[key + 9] = encoded[input[key + 9]];
            output[key + 10] = encoded[input[key + 10]];
            output[key + 11] = encoded[input[key + 11]];
            output[key + 12] = encoded[input[key + 12]];
            output[key + 13] = encoded[input[key + 13]];
            output[key + 14] = encoded[input[key + 14]];
            output[key + 15] = encoded[input[key + 15]];
        }
        for (; key < valid; ++key) output[key] = encoded[input[key]];
    } else {
        key = 0;
        for (; key + 4 <= valid; key += 4) {
            output[key] = QuantizeProbability(
                exponent[input[key]], quantizeFactor, outputZero);
            output[key + 1] = QuantizeProbability(
                exponent[input[key + 1]], quantizeFactor, outputZero);
            output[key + 2] = QuantizeProbability(
                exponent[input[key + 2]], quantizeFactor, outputZero);
            output[key + 3] = QuantizeProbability(
                exponent[input[key + 3]], quantizeFactor, outputZero);
        }
        for (; key < valid; ++key) {
            output[key] = QuantizeProbability(
                exponent[input[key]], quantizeFactor, outputZero);
        }
    }
    if (outputZero == 0) {
        memset(output + valid, 0, (size_t)(sequence - valid));
    } else {
        std::fill(output + valid, output + sequence, (uint8_t)outputZero);
    }
}

static void Eval(TFContext tfContext, TFNode node) {
    const auto info = GetNodeInfo(node);
    auto inputTensor = GetTensorByName(tfContext, info.InputNames[0]);
    auto outputTensor = GetTensorByName(tfContext, info.OutputNames[0]);
    if (GetTensorType(inputTensor) != TFCAPI_UINT8
            || GetTensorType(outputTensor) != TFCAPI_UINT8) {
        throw std::runtime_error(
            "ArmCausalMaskSoftmax requires UINT8 input and output");
    }

    const auto shape = GetTensorShape(inputTensor);
    const int sequence = ValidateShape(shape);
    const long count = GetTensorCount(inputTensor, 0);
    if (count <= 0 || count % sequence != 0) {
        throw std::runtime_error(
            "ArmCausalMaskSoftmax got inconsistent tensor element count");
    }
    const long rows = count / sequence;

    auto inputQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
    auto outputQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);
    if (!inputQuant.IsValid() || !outputQuant.IsValid()) {
        throw std::runtime_error(
            "ArmCausalMaskSoftmax requires input/output quantization info");
    }
    const auto &inputScales = GetQuantizationScale(inputQuant);
    const auto &inputZeros = GetQuantizationZeroPoint(inputQuant);
    const auto &outputScales = GetQuantizationScale(outputQuant);
    const auto &outputZeros = GetQuantizationZeroPoint(outputQuant);
    if (!((inputScales.size() == 1 || inputScales.size() == (size_t)rows)
            && inputScales.size() == inputZeros.size())) {
        throw std::runtime_error(
            "ArmCausalMaskSoftmax input qinfo count must be 1 or outer-row count");
    }
    if (outputScales.size() != 1 || outputZeros.size() != 1) {
        throw std::runtime_error(
            "ArmCausalMaskSoftmax output requires one scalar qinfo");
    }
    const float *sidecarScales = nullptr;
    if (info.InputNames.size() == 2) {
        auto scaleTensor = GetTensorByName(tfContext, info.InputNames[1]);
        if (!scaleTensor.IsValid()) {
            scaleTensor = GetParam(tfContext, info.InputNames[1]);
        }
        if (!scaleTensor.IsValid() || GetTensorType(scaleTensor) != TFCAPI_FLOAT
                || GetTensorCount(scaleTensor, 0) != rows) {
            throw std::runtime_error(
                "ArmCausalMaskSoftmax invalid row-scale sidecar");
        }
        sidecarScales = (const float *)GetTensordata(scaleTensor);
    }
    const size_t checkedScales = sidecarScales == nullptr
        ? inputScales.size() : (size_t)rows;
    for (size_t index = 0; index < checkedScales; ++index) {
        const float scale = sidecarScales == nullptr
            ? inputScales[index] : sidecarScales[index];
        if (!(scale > 0.f) || !std::isfinite(scale)) {
            throw std::runtime_error(
                "ArmCausalMaskSoftmax input qscale must be finite and positive");
        }
    }
    for (int zero : inputZeros) {
        if (zero < 0 || zero > 255) {
            throw std::runtime_error(
                "ArmCausalMaskSoftmax input zero-point is outside [0,255]");
        }
    }
    if (!(outputScales[0] > 0.f) || !std::isfinite(outputScales[0])
            || outputZeros[0] < 0 || outputZeros[0] > 255) {
        throw std::runtime_error(
            "ArmCausalMaskSoftmax output qinfo is invalid");
    }

    const uint8_t *input = (const uint8_t *)GetTensordata(inputTensor);
    uint8_t *output = (uint8_t *)GetTensordata(outputTensor);
    const float inverseOutputScale = 1.f / outputScales[0];
    const int outputZero = outputZeros[0];
    const Param *param = (const Param *)GetNodeCustomParam(node);
    const bool sharedScale = sidecarScales == nullptr
        && inputScales.size() == 1;
    const float sharedExponentRatio = sharedScale
        ? expf(-inputScales[0]) : 0.f;
#ifdef _OPENMP
    const int threads = param != nullptr && param->threads > 0
        ? param->threads : omp_get_max_threads();
    const bool shouldParallel = threads > 1
        && count >= 32 * 1024 && !omp_in_parallel();
    #pragma omp parallel for schedule(static, 8) num_threads(threads) if(shouldParallel)
#endif
    for (long row = 0; row < rows; ++row) {
        // The query axis is the penultimate axis, hence row % S for both
        // [H,S,S] and [B,H,S,S].
        const int query = (int)(row % sequence);
        const size_t qindex = inputScales.size() == 1 ? 0 : (size_t)row;
        const float rowScale = sidecarScales == nullptr
            ? inputScales[qindex] : sidecarScales[row];
        const float exponentRatio = sharedScale
            ? sharedExponentRatio : expf(-rowScale);
        ProcessRow(
            input + row * sequence,
            output + row * sequence,
            sequence,
            query, exponentRatio,
            inverseOutputScale,
            outputZero);
    }
}

static void Free(TFContext tfContext, TFNode node) {
    FreeNodeCustomParam(node, [](void *value) { delete (Param *)value; });
}

}  // namespace ArmCausalMaskSoftmax

RegistOp(ArmCausalMaskSoftmax)
.Set(
    ArmCausalMaskSoftmax::Prepare,
    ArmCausalMaskSoftmax::Reshape,
    ArmCausalMaskSoftmax::Eval,
    ArmCausalMaskSoftmax::Free);

}  // namespace TFDLOP

#endif  // NPU40T_ARM_CAUSAL_MASK_SOFTMAX_H

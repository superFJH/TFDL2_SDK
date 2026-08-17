// Explicit UINT8 -> FP32/FP16 boundary for source graphs.
//
// The built-in Op.DeQuantize currently crashes while TFContext closes a graph
// that contains a source-level DeQuantize node.  QuantizeLite intentionally
// does not insert or rewrite graph nodes, so mixed-precision graphs need a
// boundary that is valid before the weights are quantized.  This custom op
// supplies that boundary without TFContext.Modify.

#ifndef TFDL_DEQUANTIZE_LINEAR_H
#define TFDL_DEQUANTIZE_LINEAR_H

#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace TFDL_CAPI;

namespace TFDLOP {
namespace DeQuantizeLinear {

struct Param {
    bool fp16 = true;
};

static uint16_t floatToFp16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
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
        ++exponent;
        if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(
        sign | (uint16_t)(exponent << 10) | (uint16_t)(mantissa >> 13));
}

void Prepare(TFContext tfContext, TFNode node) {
    string error;
    json11::Json json = json11::Json::parse(GetNodeCustomJsonStr(node), error);
    Param *param = new Param();
    const string dst = json["dstType"].string_value();
    param->fp16 = dst.empty() || dst == "fp16" || dst == "float16";
    FreeNodeCustomParam(node, [](void *value) { delete (Param *)value; });
    NewNodeCustomParam(node, [&param]() -> void * { return param; });
}

void Reshape(TFContext tfContext, TFNode node) {
    const auto info = GetNodeInfo(node);
    TFCHECK_EQ(info.InputNames.size(), 1);
    TFCHECK_EQ(info.OutputNames.size(), 1);
    const auto input = GetTensorByName(tfContext, info.InputNames[0]);
    const auto output = GetTensorByName(tfContext, info.OutputNames[0]);
    string error;
    const json11::Json json = json11::Json::parse(
        GetNodeCustomJsonStr(node), error);
    const string dst = json["dstType"].string_value();
    const bool fp16 = dst.empty() || dst == "fp16" || dst == "float16";
    ReSizeTensor(output, GetTensorShape(input));
    SetTensorType(output, fp16 ? TFCAPI_FLOAT16 : TFCAPI_FLOAT);
}

void Eval(TFContext tfContext, TFNode node) {
    const auto info = GetNodeInfo(node);
    const auto input = GetTensorByName(tfContext, info.InputNames[0]);
    const auto output = GetTensorByName(tfContext, info.OutputNames[0]);
    TFCHECK_EQ(GetTensorType(input), TFCAPI_UINT8);
    const int count = GetTensorCount(input, 0);
    const auto quant = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
    const uint8_t *source = (const uint8_t *)GetTensordata(input);
    Param *param = (Param *)GetNodeCustomParam(node);
    if (!param->fp16) {
        DeQuantizeTensorData(
            (float *)GetTensordata(output),
            const_cast<uint8_t *>(source), count, quant);
        return;
    }
    std::vector<float> temporary((size_t)count);
    DeQuantizeTensorData(
        temporary.data(), const_cast<uint8_t *>(source), count, quant);
    uint16_t *destination = (uint16_t *)GetTensordata(output);
    for (int index = 0; index < count; ++index) {
        destination[index] = floatToFp16(temporary[(size_t)index]);
    }
}

void Free(TFContext tfContext, TFNode node) {
    FreeNodeCustomParam(node, [](void *value) { delete (Param *)value; });
}

}  // namespace DeQuantizeLinear

// Do not use a name beginning with the built-in "DeQuantize" op.  The custom
// registry treats that prefix as covered by the core op and silently skips it.
RegistOp(ExplicitDequantize)
    .Set(DeQuantizeLinear::Prepare, DeQuantizeLinear::Reshape,
         DeQuantizeLinear::Eval, DeQuantizeLinear::Free);
}  // namespace TFDLOP

#endif

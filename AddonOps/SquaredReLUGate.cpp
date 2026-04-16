//
// SquaredReLUGate 自定义算子 — 融合 relu(gate)^2 * up 操作
// 用于 Falcon-Perception 的 Gated FFN 层
//
// 输入: packed [..., 2*hidden_dim]  (gate 和 up 交织排列: [g0,u0,g1,u1,...])
// 输出: result [..., hidden_dim]    = relu(gate)^2 * up
//
// JSON 参数:
//   "hiddenDim": int, 隐藏层维度 (output 的最后一维)
//

#ifndef NPU40T_SQUARED_RELU_GATE_H
#define NPU40T_SQUARED_RELU_GATE_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <cstring>
#include <cassert>

using namespace TFDL_CAPI;

namespace TFDLOP {
    namespace SquaredReLUGate {
        struct SquaredReLUGateParam {
            int hiddenDim = 0;
        };

        void Prepare(TFContext tfContext, TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node), err);
            if (!err.empty()) {
                printf("[SquaredReLUGate] JSON parse error: %s\n", err.c_str());
            }

            SquaredReLUGateParam *p = new SquaredReLUGateParam();
            p->hiddenDim = param["hiddenDim"].int_value();

            FreeNodeCustomParam(node, [](void *cp) { delete (SquaredReLUGateParam *)cp; });
            NewNodeCustomParam(node, [&p]() -> void * { return p; });
        }

        void Reshape(TFContext tfContext, TFNode node) {
            SquaredReLUGateParam *p = (SquaredReLUGateParam *)GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            TFCHECK_EQ(info.InputNames.size(), 1);
            TFCHECK_EQ(info.OutputNames.size(), 1);

            auto inputData  = GetTensorByName(tfContext, info.InputNames[0]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);

            auto inShape = GetTensorShape(inputData);
            vector<int> outShape = inShape;
            // 最后一维从 2*hiddenDim 变为 hiddenDim
            outShape[outShape.size() - 1] = p->hiddenDim;
            ReSizeTensor(outputData, outShape);
            SetTensorType(outputData, GetTensorType(inputData));
        }

        static void squaredReluGateFloat(
            const float *packed, float *output,
            int totalRows, int hiddenDim
        ) {
            // packed: [totalRows, 2*hiddenDim]  (交织: [g0,u0,g1,u1,...])
            // output: [totalRows, hiddenDim]     = relu(gate)^2 * up
            int packedDim = 2 * hiddenDim;
            for (int r = 0; r < totalRows; r++) {
                const float *row = packed + r * packedDim;
                float *outRow = output + r * hiddenDim;
                for (int d = 0; d < hiddenDim; d++) {
                    float gate = row[2 * d];
                    float up   = row[2 * d + 1];
                    // relu(gate) = max(0, gate)
                    if (gate < 0.0f) gate = 0.0f;
                    outRow[d] = gate * gate * up;
                }
            }
        }

        void Eval(TFContext tfContext, TFNode node) {
            SquaredReLUGateParam *p = (SquaredReLUGateParam *)GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            auto inputData  = GetTensorByName(tfContext, info.InputNames[0]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);

            auto inShape = GetTensorShape(inputData);
            int lastDim = inShape[inShape.size() - 1];
            int count = GetTensorCount(inputData, 0);
            int totalRows = count / lastDim;

            if (GetTensorType(inputData) == TFCAPI_FLOAT) {
                float *in  = (float *)GetTensordata(inputData);
                float *out = (float *)GetTensordata(outputData);
                squaredReluGateFloat(in, out, totalRows, p->hiddenDim);

            } else if (GetTensorType(inputData) == TFCAPI_UINT8) {
                // uint8 路径
                uint8_t *in  = (uint8_t *)GetTensordata(inputData);
                uint8_t *out = (uint8_t *)GetTensordata(outputData);

                auto inQuant  = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
                auto outQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);

                float *inFloat  = new float[count];
                float *outFloat = new float[totalRows * p->hiddenDim];

                DeQuantizeTensorData(inFloat, in, count, inQuant);
                squaredReluGateFloat(inFloat, outFloat, totalRows, p->hiddenDim);
                QuantizeTensorData(out, outFloat, totalRows * p->hiddenDim, outQuant);

                delete[] inFloat;
                delete[] outFloat;
            }
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *cp) { delete (SquaredReLUGateParam *)cp; });
        }
    }
    RegistOp(SquaredReLUGate)
    .Set(SquaredReLUGate::Prepare, SquaredReLUGate::Reshape, SquaredReLUGate::Eval, SquaredReLUGate::Free);
}

#endif // NPU40T_SQUARED_RELU_GATE_H

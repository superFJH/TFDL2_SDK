//
// RMSNorm 自定义算子 — Root Mean Square Layer Normalization
// 公式: y = x / sqrt(mean(x^2) + eps) * gamma + beta
// 其中 gamma 和 beta 通过 JSON 配置传入（如需）
// 默认: y = x / sqrt(mean(x^2) + eps)（无仿射变换）
//
// 输入: x [..., D]
// 输出: y [..., D]（与输入同形状）
//
// JSON 参数:
//   "eps": float, 默认 1e-5
//   "normalizedShape": int, 沿最后一个维度归一化的元素数
//

#ifndef NPU40T_RMSNORM_H
#define NPU40T_RMSNORM_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <cstring>
#include <cassert>

using namespace TFDL_CAPI;

namespace TFDLOP {
    namespace RMSNorm {
        struct RMSNormParam {
            float eps = 1e-5f;
        };

        void Prepare(TFContext tfContext, TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node), err);
            if (!err.empty()) {
                printf("[RMSNorm] JSON parse error: %s\n", err.c_str());
            }

            RMSNormParam *p = new RMSNormParam();
            p->eps = (float)param["eps"].number_value();
            if (p->eps == 0.0f) p->eps = 1e-5f;

            FreeNodeCustomParam(node, [](void *cp) { delete (RMSNormParam *)cp; });
            NewNodeCustomParam(node, [&p]() -> void * { return p; });
        }

        void Reshape(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            TFCHECK_EQ(info.InputNames.size(), 1);
            TFCHECK_EQ(info.OutputNames.size(), 1);

            auto inputData  = GetTensorByName(tfContext, info.InputNames[0]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);

            ReSizeTensor(outputData, GetTensorShape(inputData));
            SetTensorType(outputData, GetTensorType(inputData));
        }

        static void rmsNormFloat(
            const float *input, float *output,
            int totalElements, int lastDim, float eps
        ) {
            // input/output 形状: [N, lastDim]，其中 N = totalElements / lastDim
            int N = totalElements / lastDim;
            for (int i = 0; i < N; i++) {
                const float *x = input + i * lastDim;
                float *y = output + i * lastDim;

                // 计算 mean(x^2)
                float sumSq = 0.0f;
                for (int d = 0; d < lastDim; d++) {
                    sumSq += x[d] * x[d];
                }
                float rms = sqrtf(sumSq / lastDim + eps);
                float invRms = 1.0f / rms;

                for (int d = 0; d < lastDim; d++) {
                    y[d] = x[d] * invRms;
                }
            }
        }

        void Eval(TFContext tfContext, TFNode node) {
            RMSNormParam *p = (RMSNormParam *)GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            auto inputData  = GetTensorByName(tfContext, info.InputNames[0]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);

            int count = GetTensorCount(inputData, 0);
            auto shape = GetTensorShape(inputData);
            int lastDim = shape[shape.size() - 1];

            if (GetTensorType(inputData) == TFCAPI_FLOAT) {
                float *in  = (float *)GetTensordata(inputData);
                float *out = (float *)GetTensordata(outputData);
                rmsNormFloat(in, out, count, lastDim, p->eps);

            } else if (GetTensorType(inputData) == TFCAPI_UINT8) {
                // uint8 路径: 反量化 → float 计算 → 重新量化
                uint8_t *in  = (uint8_t *)GetTensordata(inputData);
                uint8_t *out = (uint8_t *)GetTensordata(outputData);

                auto inQuant  = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
                auto outQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);

                float *inFloat  = new float[count];
                float *outFloat = new float[count];

                DeQuantizeTensorData(inFloat, in, count, inQuant);
                rmsNormFloat(inFloat, outFloat, count, lastDim, p->eps);
                QuantizeTensorData(out, outFloat, count, outQuant);

                delete[] inFloat;
                delete[] outFloat;
            }
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *cp) { delete (RMSNormParam *)cp; });
        }
    }
    RegistOp(RMSNorm)
    .Set(RMSNorm::Prepare, RMSNorm::Reshape, RMSNorm::Eval, RMSNorm::Free);
}

#endif // NPU40T_RMSNORM_H

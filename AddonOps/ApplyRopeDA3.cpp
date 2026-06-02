//
// DA3 2D axial rotary embedding custom op.
//
// 输入:
//   Q, K   : [B, num_heads, N, head_dim]
//   sin/cos: [1|B, 1|num_heads, HW, head_dim]
//
// 约定:
//   - 序列前缀长度 = N - HW，前缀 token 原样复制
//   - head_dim 被拆成 4 个 quarter:
//       [x_front, x_back, y_front, y_back]
//   - 分别在 x / y 两组内部做 rotate-half:
//       out_x_front = x_front * cos_x_front - x_back * sin_x_front
//       out_x_back  = x_back  * cos_x_back  + x_front * sin_x_back
//       out_y_front = y_front * cos_y_front - y_back * sin_y_front
//       out_y_back  = y_back  * cos_y_back  + y_front * sin_y_back
//

#ifndef NPU40T_APPLY_ROPE_DA3_H
#define NPU40T_APPLY_ROPE_DA3_H

#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cassert>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace TFDL_CAPI;
namespace TFDLOP {
    namespace ApplyRopeDA3 {
        struct ApplyRopeDA3Param {};

        void Prepare(TFContext tfContext, TFNode node) {
            auto *p = new ApplyRopeDA3Param();
            FreeNodeCustomParam(node, [](void *customparam) {
                delete (ApplyRopeDA3Param *)customparam;
            });
            NewNodeCustomParam(node, [&p]() -> void * {
                return p;
            });
        }

        void Reshape(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            TFCHECK_EQ(info.InputNames.size(), 4);
            TFCHECK_EQ(info.OutputNames.size(), 2);

            auto qData = GetTensorByName(tfContext, info.InputNames[0]);
            auto kData = GetTensorByName(tfContext, info.InputNames[1]);
            auto sinData = GetTensorByName(tfContext, info.InputNames[2]);
            if (!sinData.IsValid()) {
                sinData = GetParam(tfContext, info.InputNames[2]);
            }
            auto cosData = GetTensorByName(tfContext, info.InputNames[3]);
            if (!cosData.IsValid()) {
                cosData = GetParam(tfContext, info.InputNames[3]);
            }
            auto qOutData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto kOutData = GetTensorByName(tfContext, info.OutputNames[1]);

            auto qShape = GetTensorShape(qData);
            auto kShape = GetTensorShape(kData);
            auto sinShape = GetTensorShape(sinData);
            auto cosShape = GetTensorShape(cosData);

            TFCHECK_EQ(qShape.size(), 4);
            TFCHECK_EQ(qShape.size(), kShape.size());
            for (size_t i = 0; i < qShape.size(); i++) {
                TFCHECK_EQ(qShape[i], kShape[i]);
            }
            TFCHECK_EQ(sinShape.size(), cosShape.size());
            for (size_t i = 0; i < sinShape.size(); i++) {
                TFCHECK_EQ(sinShape[i], cosShape[i]);
            }

            int headDim = qShape[3];
            TFCHECK_EQ(headDim % 4, 0);
            TFCHECK_EQ(sinShape[sinShape.size() - 1], headDim);
            TFCHECK_EQ(cosShape[cosShape.size() - 1], headDim);
            int N = qShape[2];
            int hw = sinShape[sinShape.size() - 2];
            TFCHECK_GE(N, hw);

            ReSizeTensor(qOutData, qShape);
            SetTensorType(qOutData, GetTensorType(qData));
            ReSizeTensor(kOutData, kShape);
            SetTensorType(kOutData, GetTensorType(kData));
        }

        static void ropeApplyFloat(
                const float *x, float *out,
                const float *sin, const float *cos,
                int B, int numHeads, int N, int headDim,
                int sinB, int sinH, int hw) {
            int quarter = headDim / 4;
            int prefix = N - hw;

            #pragma omp parallel for collapse(2) schedule(static)
            for (int b = 0; b < B; b++) {
                for (int h = 0; h < numHeads; h++) {
                    int sb = (sinB == 1) ? 0 : b;
                    int sh = (sinH == 1) ? 0 : h;

                    if (prefix > 0) {
                        memcpy(
                            out + ((b * numHeads + h) * N) * headDim,
                            x + ((b * numHeads + h) * N) * headDim,
                            prefix * headDim * sizeof(float)
                        );
                    }

                    for (int n = prefix; n < N; n++) {
                        int sn = n - prefix;
                        const float *xPtr = x + ((b * numHeads + h) * N + n) * headDim;
                        const float *sinPtr = sin + ((sb * sinH + sh) * hw + sn) * headDim;
                        const float *cosPtr = cos + ((sb * sinH + sh) * hw + sn) * headDim;
                        float *outPtr = out + ((b * numHeads + h) * N + n) * headDim;

                        const float *x0 = xPtr;
                        const float *x1 = xPtr + quarter;
                        const float *x2 = xPtr + quarter * 2;
                        const float *x3 = xPtr + quarter * 3;
                        const float *s0 = sinPtr;
                        const float *s1 = sinPtr + quarter;
                        const float *s2 = sinPtr + quarter * 2;
                        const float *s3 = sinPtr + quarter * 3;
                        const float *c0 = cosPtr;
                        const float *c1 = cosPtr + quarter;
                        const float *c2 = cosPtr + quarter * 2;
                        const float *c3 = cosPtr + quarter * 3;
                        float *o0 = outPtr;
                        float *o1 = outPtr + quarter;
                        float *o2 = outPtr + quarter * 2;
                        float *o3 = outPtr + quarter * 3;

                        for (int d = 0; d < quarter; d++) {
                            o0[d] = x0[d] * c0[d] - x1[d] * s0[d];
                            o1[d] = x1[d] * c1[d] + x0[d] * s1[d];
                            o2[d] = x2[d] * c2[d] - x3[d] * s2[d];
                            o3[d] = x3[d] * c3[d] + x2[d] * s3[d];
                        }
                    }
                }
            }
        }

        void Eval(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            auto qData = GetTensorByName(tfContext, info.InputNames[0]);
            auto kData = GetTensorByName(tfContext, info.InputNames[1]);
            auto sinData = GetTensorByName(tfContext, info.InputNames[2]);
            if (!sinData.IsValid()) {
                sinData = GetParam(tfContext, info.InputNames[2]);
            }
            auto cosData = GetTensorByName(tfContext, info.InputNames[3]);
            if (!cosData.IsValid()) {
                cosData = GetParam(tfContext, info.InputNames[3]);
            }
            auto qOutData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto kOutData = GetTensorByName(tfContext, info.OutputNames[1]);

            auto qShape = GetTensorShape(qData);
            auto sinShape = GetTensorShape(sinData);
            int B = qShape[0];
            int numHeads = qShape[1];
            int N = qShape[2];
            int headDim = qShape[3];
            int hw = sinShape[sinShape.size() - 2];
            int sinB = sinShape[0];
            int sinH = sinShape[1];

            const float *sinPtr = (const float *)GetTensordata(sinData);
            const float *cosPtr = (const float *)GetTensordata(cosData);

            if (GetTensorType(qData) == TFCAPI_FLOAT && GetTensorType(kData) == TFCAPI_FLOAT) {
                const float *qPtr = (const float *)GetTensordata(qData);
                const float *kPtr = (const float *)GetTensordata(kData);
                float *qOutPtr = (float *)GetTensordata(qOutData);
                float *kOutPtr = (float *)GetTensordata(kOutData);
                ropeApplyFloat(qPtr, qOutPtr, sinPtr, cosPtr, B, numHeads, N, headDim, sinB, sinH, hw);
                ropeApplyFloat(kPtr, kOutPtr, sinPtr, cosPtr, B, numHeads, N, headDim, sinB, sinH, hw);
            } else if (GetTensorType(qData) == TFCAPI_UINT8 && GetTensorType(kData) == TFCAPI_UINT8) {
                int total = B * numHeads * N * headDim;
                auto qQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
                auto kQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[1]);
                float *qFloat = new float[total];
                float *kFloat = new float[total];
                float *qOutFloat = new float[total];
                float *kOutFloat = new float[total];
                DeQuantizeTensorData(qFloat, (uint8_t *)GetTensordata(qData), total, qQuant);
                DeQuantizeTensorData(kFloat, (uint8_t *)GetTensordata(kData), total, kQuant);
                ropeApplyFloat(qFloat, qOutFloat, sinPtr, cosPtr, B, numHeads, N, headDim, sinB, sinH, hw);
                ropeApplyFloat(kFloat, kOutFloat, sinPtr, cosPtr, B, numHeads, N, headDim, sinB, sinH, hw);
                auto qOutQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);
                auto kOutQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[1]);
                QuantizeTensorData((uint8_t *)GetTensordata(qOutData), qOutFloat, total, qOutQuant);
                QuantizeTensorData((uint8_t *)GetTensordata(kOutData), kOutFloat, total, kOutQuant);
                delete[] qFloat;
                delete[] kFloat;
                delete[] qOutFloat;
                delete[] kOutFloat;
            }
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *customparam) {
                delete (ApplyRopeDA3Param *)customparam;
            });
        }
    }
    RegistOp(ApplyRopeDA3)
    .Set(ApplyRopeDA3::Prepare, ApplyRopeDA3::Reshape, ApplyRopeDA3::Eval, ApplyRopeDA3::Free);
}

#endif

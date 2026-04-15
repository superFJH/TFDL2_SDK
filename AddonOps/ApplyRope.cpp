//
// Created for apply_rope custom operator - Rotary Position Embedding
//

#ifndef NPU40T_APPLY_ROPE_H
#define NPU40T_APPLY_ROPE_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <cstring>
#include <cassert>
using namespace TFDL_CAPI;
namespace TFDLOP {
    namespace ApplyRope {
        struct ApplyRopeParam {
            // No JSON parameters needed — behavior is fully determined by input tensor shapes
        };

        void Prepare(TFContext tfContext, TFNode node) {
            ApplyRopeParam *param = new ApplyRopeParam();
            FreeNodeCustomParam(node, [](void *customparam) {
                delete (ApplyRopeParam *) customparam;
            });
            NewNodeCustomParam(node, [&param]() -> void * {
                return param;
            });
        }

        void Reshape(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            TFCHECK_EQ(info.InputNames.size(), 4);
            TFCHECK_EQ(info.OutputNames.size(), 2);

            auto qData = GetTensorByName(tfContext, info.InputNames[0]);
            auto kData = GetTensorByName(tfContext, info.InputNames[1]);
            auto sinData = GetTensorByName(tfContext, info.InputNames[2]);
            auto cosData = GetTensorByName(tfContext, info.InputNames[3]);
            auto qOutData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto kOutData = GetTensorByName(tfContext, info.OutputNames[1]);

            auto qShape = GetTensorShape(qData);
            auto kShape = GetTensorShape(kData);
            auto sinShape = GetTensorShape(sinData);
            auto cosShape = GetTensorShape(cosData);

            // q and k must have same shape: [B, num_heads, N, head_dim]
            TFCHECK_EQ(qShape.size(), 4);
            TFCHECK_EQ(qShape.size(), kShape.size());
            for (size_t i = 0; i < qShape.size(); i++) {
                TFCHECK_EQ(qShape[i], kShape[i]);
            }

            // sin and cos must have same shape: [broadcast..., hw, head_dim]
            TFCHECK_EQ(sinShape.size(), cosShape.size());
            for (size_t i = 0; i < sinShape.size(); i++) {
                TFCHECK_EQ(sinShape[i], cosShape[i]);
            }

            // head_dim must match
            int headDim = qShape[3];
            TFCHECK_EQ(sinShape[sinShape.size() - 1], headDim);
            TFCHECK_EQ(cosShape[cosShape.size() - 1], headDim);

            // hw (sin's seq dimension) must be <= N (q's seq dimension)
            int N = qShape[2];
            int hw = sinShape[sinShape.size() - 2];
            TFCHECK_GE(N, hw);

            // Output shapes same as q and k
            ReSizeTensor(qOutData, qShape);
            SetTensorType(qOutData, GetTensorType(qData));
            ReSizeTensor(kOutData, kShape);
            SetTensorType(kOutData, GetTensorType(kData));
        }

        // Apply RoPE to one tensor (q or k)
        // x: [B, num_heads, N, head_dim]
        // sin/cos: [sb, sh, hw, head_dim] (with broadcasting on first dims)
        // out: [B, num_heads, N, head_dim]
        static void ropeApplyFloat(
                const float *x, float *out,
                const float *sin, const float *cos,
                int B, int numHeads, int N, int headDim,
                int sinB, int sinH, int hw) {
            int half = headDim / 2;
            int prefix = N - hw;

            for (int b = 0; b < B; b++) {
                for (int h = 0; h < numHeads; h++) {
                    // Broadcasting indices for sin/cos
                    int sb = (sinB == 1) ? 0 : b;
                    int sh = (sinH == 1) ? 0 : h;

                    // Copy prefix positions unchanged
                    if (prefix > 0) {
                        memcpy(
                            out + ((b * numHeads + h) * N) * headDim,
                            x + ((b * numHeads + h) * N) * headDim,
                            prefix * headDim * sizeof(float)
                        );
                    }

                    // Apply RoPE to positions [prefix..N-1]
                    for (int n = prefix; n < N; n++) {
                        int sn = n - prefix; // index into sin/cos seq dimension
                        const float *xPtr = x + ((b * numHeads + h) * N + n) * headDim;
                        const float *sinPtr = sin + ((sb * sinH + sh) * hw + sn) * headDim;
                        const float *cosPtr = cos + ((sb * sinH + sh) * hw + sn) * headDim;
                        float *outPtr = out + ((b * numHeads + h) * N + n) * headDim;

                        for (int d = 0; d < headDim; d++) {
                            float rotated;
                            if (d < half) {
                                rotated = -xPtr[d + half];
                            } else {
                                rotated = xPtr[d - half];
                            }
                            outPtr[d] = xPtr[d] * cosPtr[d] + rotated * sinPtr[d];
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
            auto cosData = GetTensorByName(tfContext, info.InputNames[3]);
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

            // sin and cos are always float
            const float *sinPtr = (const float *) GetTensordata(sinData);
            const float *cosPtr = (const float *) GetTensordata(cosData);

            if (GetTensorType(qData) == TFCAPI_FLOAT && GetTensorType(kData) == TFCAPI_FLOAT) {
                // Pure float path
                const float *qPtr = (const float *) GetTensordata(qData);
                const float *kPtr = (const float *) GetTensordata(kData);
                float *qOutPtr = (float *) GetTensordata(qOutData);
                float *kOutPtr = (float *) GetTensordata(kOutData);

                ropeApplyFloat(qPtr, qOutPtr, sinPtr, cosPtr, B, numHeads, N, headDim, sinB, sinH, hw);
                ropeApplyFloat(kPtr, kOutPtr, sinPtr, cosPtr, B, numHeads, N, headDim, sinB, sinH, hw);

            } else if (GetTensorType(qData) == TFCAPI_UINT8 && GetTensorType(kData) == TFCAPI_UINT8) {
                // Uint8 path: dequantize -> float RoPE -> requantize
                int totalElements = B * numHeads * N * headDim;
                auto qQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
                auto kQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[1]);

                // Dequantize q and k to float
                float *qFloat = new float[totalElements];
                float *kFloat = new float[totalElements];
                DeQuantizeTensorData(qFloat, (uint8_t *) GetTensordata(qData), totalElements, qQuant);
                DeQuantizeTensorData(kFloat, (uint8_t *) GetTensordata(kData), totalElements, kQuant);

                // Apply RoPE in float
                float *qOutFloat = new float[totalElements];
                float *kOutFloat = new float[totalElements];
                ropeApplyFloat(qFloat, qOutFloat, sinPtr, cosPtr, B, numHeads, N, headDim, sinB, sinH, hw);
                ropeApplyFloat(kFloat, kOutFloat, sinPtr, cosPtr, B, numHeads, N, headDim, sinB, sinH, hw);

                // Requantize back to uint8
                auto qOutQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);
                auto kOutQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[1]);
                QuantizeTensorData((uint8_t *) GetTensordata(qOutData), qOutFloat, totalElements, qOutQuant);
                QuantizeTensorData((uint8_t *) GetTensordata(kOutData), kOutFloat, totalElements, kOutQuant);

                delete[] qFloat;
                delete[] kFloat;
                delete[] qOutFloat;
                delete[] kOutFloat;
            }
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *customparam) {
                delete (ApplyRopeParam *) customparam;
            });
        }
    }
    RegistOp(ApplyRope)
    .Set(ApplyRope::Prepare, ApplyRope::Reshape, ApplyRope::Eval, ApplyRope::Free);
}

#endif //NPU40T_APPLY_ROPE_H

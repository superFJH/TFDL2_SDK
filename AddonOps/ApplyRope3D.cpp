//
// ApplyRope3D 自定义算子 — 三维混合旋转位置编码 (1D 时间 + 2D Golden-Gate 空间)
// 用于 Falcon-Perception 模型的注意力机制
//
// 头维度被分为两半:
//   前半部分: 标准 1D RoPE (时间维度)
//   后半部分: 2D Golden-Gate RoPE (空间维度, 学习频率)
//
// 输入:
//   [0] Q:    [B, num_heads, S, head_dim]    查询张量
//   [1] K:    [B, num_kv_heads, S, head_dim] 键张量
//   [2] sin1d: [B, S, half_dim] 或 [1, 1, S, half_dim]  1D sin 频率
//   [3] cos1d: [B, S, half_dim] 或 [1, 1, S, half_dim]  1D cos 频率
//   [4] sin2d: [B, S, num_heads, quarter_dim]  2D sin 频率 (可选, 用于 prefill)
//   [5] cos2d: [B, S, num_heads, quarter_dim]  2D cos 频率 (可选, 用于 prefill)
//
// 输出:
//   [0] Q_rotated: [B, num_heads, S, head_dim]
//   [1] K_rotated: [B, num_kv_heads, S, head_dim]
//
// JSON 参数:
//   "apply2d": bool, 是否应用 2D RoPE (prefill 时为 true, decode 时为 false)
//

#ifndef NPU40T_APPLY_ROPE3D_H
#define NPU40T_APPLY_ROPE3D_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <cstring>
#include <cassert>

using namespace TFDL_CAPI;

namespace TFDLOP {
    namespace ApplyRope3D {
        struct ApplyRope3DParam {
            bool apply2d = false;
        };

        void Prepare(TFContext tfContext, TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node), err);
            if (!err.empty()) {
                printf("[ApplyRope3D] JSON parse error: %s\n", err.c_str());
            }

            ApplyRope3DParam *p = new ApplyRope3DParam();
            p->apply2d = param["apply2d"].bool_value();

            FreeNodeCustomParam(node, [](void *cp) { delete (ApplyRope3DParam *)cp; });
            NewNodeCustomParam(node, [&p]() -> void * { return p; });
        }

        void Reshape(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            int nInputs = (int)info.InputNames.size();
            TFCHECK_GE(nInputs, 4);  // 至少需要 Q, K, sin1d, cos1d
            if (nInputs > 4) {
                TFCHECK_EQ(nInputs, 6);  // 有 2D 时必须是 6 个输入
            }
            TFCHECK_EQ(info.OutputNames.size(), 2);

            auto qData = GetTensorByName(tfContext, info.InputNames[0]);
            auto kData = GetTensorByName(tfContext, info.InputNames[1]);
            auto qOut  = GetTensorByName(tfContext, info.OutputNames[0]);
            auto kOut  = GetTensorByName(tfContext, info.OutputNames[1]);

            ReSizeTensor(qOut, GetTensorShape(qData));
            SetTensorType(qOut, GetTensorType(qData));
            ReSizeTensor(kOut, GetTensorShape(kData));
            SetTensorType(kOut, GetTensorType(kData));
        }

        // 1D RoPE 应用 (标准旋转位置编码)
        // x: [B, H, S, D_half]
        // sin/cos: [B, S, D_half] 或 [1, 1, S, D_half]
        static void applyRope1DFloat(
            const float *x, float *out,
            const float *sin, const float *cos,
            int B, int H, int S, int halfDim,
            int sinB, int sinH, int sinS
        ) {
            int quarterDim = halfDim / 2;
            for (int b = 0; b < B; b++) {
                for (int h = 0; h < H; h++) {
                    for (int s = 0; s < S; s++) {
                        int sb = (sinB == 1) ? 0 : b;
                        int sh = (sinH == 1) ? 0 : h;
                        int ss = s;  // sin 的序列维度与输入对齐

                        const float *xPtr    = x   + ((b * H + h) * S + s) * halfDim;
                        const float *sinPtr  = sin + ((sb * sinH + sh) * sinS + ss) * halfDim;
                        const float *cosPtr  = cos + ((sb * sinH + sh) * sinS + ss) * halfDim;
                        float *outPtr = out + ((b * H + h) * S + s) * halfDim;

                        // 将 halfDim 视为 complex pairs: (x[2i], x[2i+1]) * (cos[i], sin[i])
                        for (int d = 0; d < halfDim; d += 2) {
                            float x0 = xPtr[d];
                            float x1 = xPtr[d + 1];
                            float c = cosPtr[d];
                            float s_val = sinPtr[d];
                            outPtr[d]     = x0 * c - x1 * s_val;
                            outPtr[d + 1] = x0 * s_val + x1 * c;
                        }
                    }
                }
            }
        }

        // 2D Golden-Gate RoPE 应用
        // x: [B, H, S, D_half]
        // sin2d/cos2d: [B, S, H, quarterDim] — 预计算的角度
        static void applyRope2DFloat(
            const float *x, float *out,
            const float *sin2d, const float *cos2d,
            int B, int H, int S, int halfDim,
            int sinB, int sinS, int sinH, int quarterDim
        ) {
            // 2D RoPE 作用于 head_dim 的后半部分
            // x 的排列: [..., 0::2] 和 [..., 1::2] 交替对
            for (int b = 0; b < B; b++) {
                for (int h = 0; h < H; h++) {
                    for (int s = 0; s < S; s++) {
                        const float *xPtr = x + ((b * H + h) * S + s) * halfDim;
                        float *outPtr = out + ((b * H + h) * S + s) * halfDim;

                        int sb = (sinB == 1) ? 0 : b;
                        int ss = s;
                        int sh = h;

                        // sin2d/cos2d 形状: [B, S, H, quarterDim]
                        const float *sinPtr = sin2d + ((sb * sinS + ss) * sinH + sh) * quarterDim;
                        const float *cosPtr = cos2d + ((sb * sinS + ss) * sinH + sh) * quarterDim;

                        // 将 halfDim 个元素视为 quarterDim 个 complex pairs
                        // x_even = x[..., 0::2], x_odd = x[..., 1::2]
                        for (int d = 0; d < quarterDim; d++) {
                            float x_even = xPtr[2 * d];
                            float x_odd  = xPtr[2 * d + 1];
                            float c = cosPtr[d];
                            float s_val = sinPtr[d];
                            outPtr[2 * d]     = x_even * c - x_odd * s_val;
                            outPtr[2 * d + 1] = x_even * s_val + x_odd * c;
                        }
                    }
                }
            }
        }

        void Eval(TFContext tfContext, TFNode node) {
            ApplyRope3DParam *p = (ApplyRope3DParam *)GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);

            auto qData  = GetTensorByName(tfContext, info.InputNames[0]);
            auto kData  = GetTensorByName(tfContext, info.InputNames[1]);
            auto sin1d  = GetTensorByName(tfContext, info.InputNames[2]);
            auto cos1d  = GetTensorByName(tfContext, info.InputNames[3]);
            auto qOut   = GetTensorByName(tfContext, info.OutputNames[0]);
            auto kOut   = GetTensorByName(tfContext, info.OutputNames[1]);

            auto qShape = GetTensorShape(qData);
            int B = qShape[0], nHeadsQ = qShape[1], S = qShape[2], headDim = qShape[3];
            int halfDim = headDim / 2;

            auto kShape = GetTensorShape(kData);
            int nHeadsKV = kShape[1];

            auto sinShape = GetTensorShape(sin1d);
            int sinB = sinShape[0], sinH = sinShape[1], sinS = sinShape[2];

            const float *sin1dPtr = (const float *)GetTensordata(sin1d);
            const float *cos1dPtr = (const float *)GetTensordata(cos1d);

            if (GetTensorType(qData) == TFCAPI_FLOAT) {
                const float *qPtr = (const float *)GetTensordata(qData);
                const float *kPtr = (const float *)GetTensordata(kData);
                float *qOutPtr = (float *)GetTensordata(qOut);
                float *kOutPtr = (float *)GetTensordata(kOut);

                // 分配临时缓冲区用于前半/后半分离
                int qTotal = B * nHeadsQ * S;
                int kTotal = B * nHeadsKV * S;

                float *qFirstHalf  = new float[qTotal * halfDim];
                float *qSecondHalf = new float[qTotal * halfDim];
                float *kFirstHalf  = new float[kTotal * halfDim];
                float *kSecondHalf = new float[kTotal * halfDim];

                // 分离 Q 和 K 的前后半部分
                for (int i = 0; i < qTotal; i++) {
                    for (int d = 0; d < headDim; d++) {
                        if (d < halfDim)
                            qFirstHalf[i * halfDim + d] = qPtr[i * headDim + d];
                        else
                            qSecondHalf[i * halfDim + (d - halfDim)] = qPtr[i * headDim + d];
                    }
                }
                for (int i = 0; i < kTotal; i++) {
                    for (int d = 0; d < headDim; d++) {
                        if (d < halfDim)
                            kFirstHalf[i * halfDim + d] = kPtr[i * headDim + d];
                        else
                            kSecondHalf[i * halfDim + (d - halfDim)] = kPtr[i * headDim + d];
                    }
                }

                // 临时输出缓冲区
                float *qFirstOut  = new float[qTotal * halfDim];
                float *qSecondOut = new float[qTotal * halfDim];
                float *kFirstOut  = new float[kTotal * halfDim];
                float *kSecondOut = new float[kTotal * halfDim];

                // 应用 1D RoPE 到前半部分 (Q 和 K)
                applyRope1DFloat(qFirstHalf, qFirstOut, sin1dPtr, cos1dPtr,
                                 B, nHeadsQ, S, halfDim, sinB, sinH, sinS);
                applyRope1DFloat(kFirstHalf, kFirstOut, sin1dPtr, cos1dPtr,
                                 B, nHeadsKV, S, halfDim, sinB, sinH, sinS);

                // 应用 2D RoPE 到后半部分 (如果提供)
                if (p->apply2d && info.InputNames.size() >= 6) {
                    auto sin2dData = GetTensorByName(tfContext, info.InputNames[4]);
                    auto cos2dData = GetTensorByName(tfContext, info.InputNames[5]);
                    const float *sin2dPtr = (const float *)GetTensordata(sin2dData);
                    const float *cos2dPtr = (const float *)GetTensordata(cos2dData);

                    auto sin2dShape = GetTensorShape(sin2dData);
                    int s2dB = sin2dShape[0], s2dS = sin2dShape[1];
                    int s2dH = sin2dShape[2], quarterDim = sin2dShape[3];

                    applyRope2DFloat(qSecondHalf, qSecondOut, sin2dPtr, cos2dPtr,
                                     B, nHeadsQ, S, halfDim, s2dB, s2dS, s2dH, quarterDim);
                    applyRope2DFloat(kSecondHalf, kSecondOut, sin2dPtr, cos2dPtr,
                                     B, nHeadsKV, S, halfDim, s2dB, s2dS, s2dH, quarterDim);
                } else {
                    // 不应用 2D RoPE，直接复制后半部分
                    memcpy(qSecondOut, qSecondHalf, qTotal * halfDim * sizeof(float));
                    memcpy(kSecondOut, kSecondHalf, kTotal * halfDim * sizeof(float));
                }

                // 合并前后半部分到输出
                for (int i = 0; i < qTotal; i++) {
                    for (int d = 0; d < headDim; d++) {
                        if (d < halfDim)
                            qOutPtr[i * headDim + d] = qFirstOut[i * halfDim + d];
                        else
                            qOutPtr[i * headDim + d] = qSecondOut[i * halfDim + (d - halfDim)];
                    }
                }
                for (int i = 0; i < kTotal; i++) {
                    for (int d = 0; d < headDim; d++) {
                        if (d < halfDim)
                            kOutPtr[i * headDim + d] = kFirstOut[i * halfDim + d];
                        else
                            kOutPtr[i * headDim + d] = kSecondOut[i * halfDim + (d - halfDim)];
                    }
                }

                delete[] qFirstHalf;  delete[] qSecondHalf;
                delete[] kFirstHalf;  delete[] kSecondHalf;
                delete[] qFirstOut;   delete[] qSecondOut;
                delete[] kFirstOut;   delete[] kSecondOut;

            } else if (GetTensorType(qData) == TFCAPI_UINT8) {
                // uint8 路径: 反量化 → float 计算 → 重新量化
                int qCount = B * nHeadsQ * S * headDim;
                int kCount = B * nHeadsKV * S * headDim;

                auto qQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
                auto kQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[1]);

                float *qFloat = new float[qCount];
                float *kFloat = new float[kCount];
                DeQuantizeTensorData(qFloat, (uint8_t *)GetTensordata(qData), qCount, qQuant);
                DeQuantizeTensorData(kFloat, (uint8_t *)GetTensordata(kData), kCount, kQuant);

                // 创建临时 float 输入/输出（复用 float 逻辑）
                // 这里简化处理: 将 uint8 当做 float 路径处理
                float *qOutFloat = new float[qCount];
                float *kOutFloat = new float[kCount];

                // 注意: 对于 uint8 路径，需要将 Q 和 K 的 shape 信息构建出来
                // 简化实现: 仅应用 1D RoPE
                int totalQ = B * nHeadsQ * S;
                int totalK = B * nHeadsKV * S;

                float *qFirstHalf  = new float[totalQ * halfDim];
                float *qSecondHalf = new float[totalQ * halfDim];
                float *kFirstHalf  = new float[totalK * halfDim];
                float *kSecondHalf = new float[totalK * halfDim];
                float *qFirstOut   = new float[totalQ * halfDim];
                float *kFirstOut   = new float[totalK * halfDim];

                for (int i = 0; i < totalQ; i++) {
                    for (int d = 0; d < headDim; d++) {
                        if (d < halfDim)
                            qFirstHalf[i * halfDim + d] = qFloat[i * headDim + d];
                        else
                            qSecondHalf[i * halfDim + (d - halfDim)] = qFloat[i * headDim + d];
                    }
                }
                for (int i = 0; i < totalK; i++) {
                    for (int d = 0; d < headDim; d++) {
                        if (d < halfDim)
                            kFirstHalf[i * halfDim + d] = kFloat[i * headDim + d];
                        else
                            kSecondHalf[i * halfDim + (d - halfDim)] = kFloat[i * headDim + d];
                    }
                }

                applyRope1DFloat(qFirstHalf, qFirstOut, sin1dPtr, cos1dPtr,
                                 B, nHeadsQ, S, halfDim, sinB, sinH, sinS);
                applyRope1DFloat(kFirstHalf, kFirstOut, sin1dPtr, cos1dPtr,
                                 B, nHeadsKV, S, halfDim, sinB, sinH, sinS);

                // 合并
                for (int i = 0; i < totalQ; i++) {
                    for (int d = 0; d < headDim; d++) {
                        if (d < halfDim)
                            qOutFloat[i * headDim + d] = qFirstOut[i * halfDim + d];
                        else
                            qOutFloat[i * headDim + d] = qSecondHalf[i * halfDim + (d - halfDim)];
                    }
                }
                for (int i = 0; i < totalK; i++) {
                    for (int d = 0; d < headDim; d++) {
                        if (d < halfDim)
                            kOutFloat[i * headDim + d] = kFirstOut[i * halfDim + d];
                        else
                            kOutFloat[i * headDim + d] = kSecondHalf[i * halfDim + (d - halfDim)];
                    }
                }

                auto qOutQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);
                auto kOutQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[1]);
                QuantizeTensorData((uint8_t *)GetTensordata(qOut), qOutFloat, qCount, qOutQuant);
                QuantizeTensorData((uint8_t *)GetTensordata(kOut), kOutFloat, kCount, kOutQuant);

                delete[] qFloat;      delete[] kFloat;
                delete[] qOutFloat;   delete[] kOutFloat;
                delete[] qFirstHalf;  delete[] qSecondHalf;
                delete[] kFirstHalf;  delete[] kSecondHalf;
                delete[] qFirstOut;   delete[] kFirstOut;
            }
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *cp) { delete (ApplyRope3DParam *)cp; });
        }
    }
    RegistOp(ApplyRope3D)
    .Set(ApplyRope3D::Prepare, ApplyRope3D::Reshape, ApplyRope3D::Eval, ApplyRope3D::Free);
}

#endif // NPU40T_APPLY_ROPE3D_H

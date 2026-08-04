//
// Created for apply_rope custom operator - Rotary Position Embedding
//
// RoPE 公式:
//   前缀 tokens (CLS + registers) 保持不变
//   对 patch tokens 的每个位置:
//     out[d]         = x[d] * cos[d] + (-x[d+half]) * sin[d]    (d < half)
//     out[d]         = x[d] * cos[d] + x[d-half]   * sin[d]     (d >= half)
//   等价于:
//     out_front[d]   = x_front[d] * cos_front[d] - x_back[d]  * sin_front[d]
//     out_back[d]    = x_back[d]  * cos_back[d]  + x_front[d] * sin_back[d]
//

#ifndef NPU40T_APPLY_ROPE_H
#define NPU40T_APPLY_ROPE_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <cstring>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

using namespace TFDL_CAPI;
namespace TFDLOP {
    namespace ApplyRope {
        struct ApplyRopeParam {
            bool useFp16 = false;  // 使用 FP16 NEON 加速 (精度略降, 速度更快)
            bool interleaved = false;  // Adjacent complex pairs, used by MoonViT.
        };

        void Prepare(TFContext tfContext, TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node), err);

            ApplyRopeParam *p = new ApplyRopeParam();
            if (err.empty()) {
                p->useFp16 = param["useFp16"].bool_value();
                p->interleaved = param["interleaved"].bool_value();
            }

            FreeNodeCustomParam(node, [](void *customparam) {
                delete (ApplyRopeParam *) customparam;
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
            if(!sinData.IsValid()){
                sinData = GetParam(tfContext, info.InputNames[2]);
            }
            auto cosData = GetTensorByName(tfContext, info.InputNames[3]);
            if(!cosData.IsValid()){
                cosData = GetParam(tfContext, info.InputNames[3]);
            }
            auto qOutData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto kOutData = GetTensorByName(tfContext, info.OutputNames[1]);

            auto qShape = GetTensorShape(qData);
            auto kShape = GetTensorShape(kData);
            auto sinShape = GetTensorShape(sinData);
            auto cosShape = GetTensorShape(cosData);

            // q/k are [B, heads, N, head_dim].  ViT uses equal head counts,
            // while Qwen-style GQA uses more query heads than KV heads.
            TFCHECK_EQ(qShape.size(), 4);
            TFCHECK_EQ(kShape.size(), 4);
            TFCHECK_EQ(qShape[0], kShape[0]);
            TFCHECK_EQ(qShape[2], kShape[2]);
            TFCHECK_EQ(qShape[3], kShape[3]);

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

            // Output shapes follow each input independently.
            ReSizeTensor(qOutData, qShape);
            SetTensorType(qOutData, GetTensorType(qData));
            ReSizeTensor(kOutData, kShape);
            SetTensorType(kOutData, GetTensorType(kData));
        }

        // ====================================================================
        // 标量实现 (fallback, 用于非 aarch64 平台)
        // ====================================================================
        static void ropeApplyScalar(
                const float *x, float *out,
                const float *sin, const float *cos,
                int B, int numHeads, int N, int headDim,
                int sinB, int sinH, int hw, bool interleaved = false) {
            int half = headDim / 2;
            int prefix = N - hw;

            #pragma omp parallel for collapse(2) schedule(static)
            for (int b = 0; b < B; b++) {
                for (int h = 0; h < numHeads; h++) {
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
                        int sn = n - prefix;
                        const float *xPtr = x + ((b * numHeads + h) * N + n) * headDim;
                        const float *sinPtr = sin + ((sb * sinH + sh) * hw + sn) * headDim;
                        const float *cosPtr = cos + ((sb * sinH + sh) * hw + sn) * headDim;
                        float *outPtr = out + ((b * numHeads + h) * N + n) * headDim;

                        if (interleaved) {
                            for (int d = 0; d + 1 < headDim; d += 2) {
                                outPtr[d] = xPtr[d] * cosPtr[d] - xPtr[d + 1] * sinPtr[d];
                                outPtr[d + 1] = xPtr[d + 1] * cosPtr[d + 1] + xPtr[d] * sinPtr[d + 1];
                            }
                        } else {
                            // 前半: out[d] = x[d]*cos[d] - x[d+half]*sin[d]
                            for (int d = 0; d < half; d++) {
                                outPtr[d] = xPtr[d] * cosPtr[d] - xPtr[d + half] * sinPtr[d];
                            }
                            // 后半: out[d] = x[d]*cos[d] + x[d-half]*sin[d]
                            for (int d = half; d < headDim; d++) {
                                outPtr[d] = xPtr[d] * cosPtr[d] + xPtr[d - half] * sinPtr[d];
                            }
                        }
                    }
                }
            }
        }

        // ====================================================================
        // NEON 实现 (aarch64)
        //
        // 将 headDim 维度拆为 front (0..half) 和 back (half..headDim) 两段:
        //   out_front[d] = x_front[d] * cos_front[d] - x_back[d] * sin_front[d]
        //   out_back[d]  = x_back[d]  * cos_back[d]  + x_front[d] * sin_back[d]
        //
        // 每次处理 4 个 float (float32x4_t)，用 vmlaq/vmlsq 完成乘加/乘减。
        // ====================================================================
#ifdef __aarch64__
        static void ropeApplyNeon(
                const float *x, float *out,
                const float *sin, const float *cos,
                int B, int numHeads, int N, int headDim,
                int sinB, int sinH, int hw) {
            int half = headDim / 2;
            int prefix = N - hw;
            const int laneBytes = 4;  // sizeof(float)

            #pragma omp parallel for collapse(2) schedule(static)
            for (int b = 0; b < B; b++) {
                for (int h = 0; h < numHeads; h++) {
                    int sb = (sinB == 1) ? 0 : b;
                    int sh = (sinH == 1) ? 0 : h;

                    int rowStride = (b * numHeads + h) * N;
                    int scStride = (sb * sinH + sh) * hw;

                    // ---- prefix memcpy ----
                    if (prefix > 0) {
                        memcpy(out + rowStride * headDim,
                               x + rowStride * headDim,
                               prefix * headDim * sizeof(float));
                    }

                    // ---- RoPE on patch positions ----
                    for (int n = prefix; n < N; n++) {
                        int sn = n - prefix;
                        const float *xFront = x   + (rowStride + n) * headDim;
                        const float *xBack  = x   + (rowStride + n) * headDim + half;
                        const float *sinF   = sin + (scStride + sn) * headDim;
                        const float *sinBk  = sin + (scStride + sn) * headDim + half;
                        const float *cosF   = cos + (scStride + sn) * headDim;
                        const float *cosBk  = cos + (scStride + sn) * headDim + half;
                        float *outFront     = out + (rowStride + n) * headDim;
                        float *outBack      = out + (rowStride + n) * headDim + half;

                        int d = 0;
                        // ---- NEON 向量化: 每次处理 4 个 float ----
                        for (; d + 4 <= half; d += 4) {
                            float32x4_t vxF   = vld1q_f32(xFront + d);
                            float32x4_t vxB   = vld1q_f32(xBack  + d);
                            float32x4_t vcosF = vld1q_f32(cosF   + d);
                            float32x4_t vcosB = vld1q_f32(cosBk  + d);
                            float32x4_t vsinF = vld1q_f32(sinF   + d);
                            float32x4_t vsinB = vld1q_f32(sinBk  + d);

                            // out_front = xF*cosF - xB*sinF
                            float32x4_t vOutF = vmulq_f32(vxF, vcosF);
                            vOutF = vmlsq_f32(vOutF, vxB, vsinF);
                            vst1q_f32(outFront + d, vOutF);

                            // out_back = xB*cosB + xF*sinB
                            float32x4_t vOutB = vmulq_f32(vxB, vcosB);
                            vOutB = vmlaq_f32(vOutB, vxF, vsinB);
                            vst1q_f32(outBack + d, vOutB);
                        }

                        // ---- 剩余元素标量处理 ----
                        for (; d < half; d++) {
                            outFront[d] = xFront[d] * cosF[d] - xBack[d] * sinF[d];
                            outBack[d]  = xBack[d]  * cosBk[d] + xFront[d] * sinBk[d];
                        }
                    }
                }
            }
        }

        // ====================================================================
        // Float16 NEON 实现 (aarch64 with ARMv8.2-A FP16 支持)
        //
        // 策略: sin/cos 预转换为 FP16 (仅一次),
        //       每行 x 的 front/back 转为 FP16, 做 8-lane RoPE, 再转回 float32。
        //       尾部 (<8 元素) 回退到 float32 NEON + 标量。
        //
        // 相比 float32 NEON (4-lane): 数据吞吐翻倍, 代价是 f32↔f16 转换开销。
        // 对较大 headDim (如 64, 128) 有明显加速。
        // ====================================================================
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        static void ropeApplyFp16Neon(
                const float *x, float *out,
                const float *sin, const float *cos,
                int B, int numHeads, int N, int headDim,
                int sinB, int sinH, int hw) {
            int half = headDim / 2;
            int prefix = N - hw;

            // ---- 预转换 sin/cos 到 FP16 (仅做一次) ----
            int scTotal = sinB * sinH * hw * headDim;
            __fp16 *sinFp16 = new __fp16[scTotal];
            __fp16 *cosFp16 = new __fp16[scTotal];
            {
                int i = 0;
                for (; i + 8 <= scTotal; i += 8) {
                    float32x4_t sLo = vld1q_f32(sin + i);
                    float32x4_t sHi = vld1q_f32(sin + i + 4);
                    vst1q_f16(sinFp16 + i,
                              vcombine_f16(vcvt_f16_f32(sLo), vcvt_f16_f32(sHi)));
                    float32x4_t cLo = vld1q_f32(cos + i);
                    float32x4_t cHi = vld1q_f32(cos + i + 4);
                    vst1q_f16(cosFp16 + i,
                              vcombine_f16(vcvt_f16_f32(cLo), vcvt_f16_f32(cHi)));
                }
                for (; i < scTotal; i++) {
                    sinFp16[i] = (__fp16)sin[i];
                    cosFp16[i] = (__fp16)cos[i];
                }
            }

            // ---- 主循环 (OpenMP 并行化) ----
            #pragma omp parallel for collapse(2) schedule(static)
            for (int b = 0; b < B; b++) {
                for (int h = 0; h < numHeads; h++) {
                    int sb = (sinB == 1) ? 0 : b;
                    int sh = (sinH == 1) ? 0 : h;
                    int rowStride = (b * numHeads + h) * N;
                    int scStride  = (sb * sinH + sh) * hw;

                    // ---- prefix memcpy ----
                    if (prefix > 0) {
                        memcpy(out + rowStride * headDim,
                               x  + rowStride * headDim,
                               prefix * headDim * sizeof(float));
                    }

                    // ---- RoPE on patch positions ----
                    for (int n = prefix; n < N; n++) {
                        int sn = n - prefix;
                        const float *xFront = x   + (rowStride + n) * headDim;
                        const float *xBack  = x   + (rowStride + n) * headDim + half;
                        const __fp16 *sinF  = sinFp16 + (scStride + sn) * headDim;
                        const __fp16 *sinBk = sinFp16 + (scStride + sn) * headDim + half;
                        const __fp16 *cosF  = cosFp16 + (scStride + sn) * headDim;
                        const __fp16 *cosBk = cosFp16 + (scStride + sn) * headDim + half;
                        float *outFront     = out + (rowStride + n) * headDim;
                        float *outBack      = out + (rowStride + n) * headDim + half;

                        // 用于尾部回退的 float 指针
                        const float *sinF_f  = sin + (scStride + sn) * headDim;
                        const float *sinBk_f = sin + (scStride + sn) * headDim + half;
                        const float *cosF_f  = cos + (scStride + sn) * headDim;
                        const float *cosBk_f = cos + (scStride + sn) * headDim + half;

                        int d = 0;
                        // ---- FP16 向量化: 每次处理 8 个元素 ----
                        for (; d + 8 <= half; d += 8) {
                            // f32 -> f16 (两批 4 个)
                            float16x8_t vxF = vcombine_f16(
                                vcvt_f16_f32(vld1q_f32(xFront + d)),
                                vcvt_f16_f32(vld1q_f32(xFront + d + 4)));
                            float16x8_t vxB = vcombine_f16(
                                vcvt_f16_f32(vld1q_f32(xBack  + d)),
                                vcvt_f16_f32(vld1q_f32(xBack  + d + 4)));

                            // 加载预转换的 FP16 sin/cos
                            float16x8_t vsinF = vld1q_f16(sinF  + d);
                            float16x8_t vsinB = vld1q_f16(sinBk + d);
                            float16x8_t vcosF = vld1q_f16(cosF  + d);
                            float16x8_t vcosB = vld1q_f16(cosBk + d);

                            // out_front = xF*cosF - xB*sinF
                            float16x8_t vOutF = vmulq_f16(vxF, vcosF);
                            vOutF = vfmsq_f16(vOutF, vxB, vsinF);

                            // out_back = xB*cosB + xF*sinB
                            float16x8_t vOutB = vmulq_f16(vxB, vcosB);
                            vOutB = vfmaq_f16(vOutB, vxF, vsinB);

                            // f16 -> f32 并存储
                            vst1q_f32(outFront + d,     vcvt_f32_f16(vget_low_f16(vOutF)));
                            vst1q_f32(outFront + d + 4, vcvt_f32_f16(vget_high_f16(vOutF)));
                            vst1q_f32(outBack  + d,     vcvt_f32_f16(vget_low_f16(vOutB)));
                            vst1q_f32(outBack  + d + 4, vcvt_f32_f16(vget_high_f16(vOutB)));
                        }

                        // ---- 剩余 4 个一组: 使用 float32 NEON ----
                        for (; d + 4 <= half; d += 4) {
                            float32x4_t vxF32   = vld1q_f32(xFront + d);
                            float32x4_t vxB32   = vld1q_f32(xBack  + d);
                            float32x4_t vcosF32 = vld1q_f32(cosF_f  + d);
                            float32x4_t vcosB32 = vld1q_f32(cosBk_f + d);
                            float32x4_t vsinF32 = vld1q_f32(sinF_f  + d);
                            float32x4_t vsinB32 = vld1q_f32(sinBk_f + d);

                            float32x4_t vOutF32 = vmulq_f32(vxF32, vcosF32);
                            vOutF32 = vmlsq_f32(vOutF32, vxB32, vsinF32);
                            vst1q_f32(outFront + d, vOutF32);

                            float32x4_t vOutB32 = vmulq_f32(vxB32, vcosB32);
                            vOutB32 = vmlaq_f32(vOutB32, vxF32, vsinB32);
                            vst1q_f32(outBack + d, vOutB32);
                        }

                        // ---- 最终标量尾 ----
                        for (; d < half; d++) {
                            outFront[d] = xFront[d] * cosF_f[d] - xBack[d] * sinF_f[d];
                            outBack[d]  = xBack[d]  * cosBk_f[d] + xFront[d] * sinBk_f[d];
                        }
                    }
                }
            }

            delete[] sinFp16;
            delete[] cosFp16;
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif // __aarch64__

        // ====================================================================
        // 统一入口: 根据 useFp16 选择实现路径
        // ====================================================================
        static void ropeApplyFloat(
                const float *x, float *out,
                const float *sin, const float *cos,
                int B, int numHeads, int N, int headDim,
                int sinB, int sinH, int hw, bool useFp16, bool interleaved = false) {
            if (interleaved) {
                ropeApplyScalar(x, out, sin, cos, B, numHeads, N, headDim, sinB, sinH, hw, true);
                return;
            }
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            if (useFp16) {
                ropeApplyFp16Neon(x, out, sin, cos, B, numHeads, N, headDim, sinB, sinH, hw);
            } else {
                ropeApplyNeon(x, out, sin, cos, B, numHeads, N, headDim, sinB, sinH, hw);
            }
#elif defined(__aarch64__)
            (void)useFp16;
            ropeApplyNeon(x, out, sin, cos, B, numHeads, N, headDim, sinB, sinH, hw);
#else
            (void)useFp16;
            ropeApplyScalar(x, out, sin, cos, B, numHeads, N, headDim, sinB, sinH, hw);
#endif
        }

        void Eval(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            auto param = (ApplyRopeParam *)GetNodeCustomParam(node);
            auto qData = GetTensorByName(tfContext, info.InputNames[0]);
            auto kData = GetTensorByName(tfContext, info.InputNames[1]);
            auto sinData = GetTensorByName(tfContext, info.InputNames[2]);
            if(!sinData.IsValid()){
                sinData = GetParam(tfContext, info.InputNames[2]);
            }
            auto cosData = GetTensorByName(tfContext, info.InputNames[3]);
            if(!cosData.IsValid()){
                cosData = GetParam(tfContext, info.InputNames[3]);
            }
            auto qOutData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto kOutData = GetTensorByName(tfContext, info.OutputNames[1]);

            auto qShape = GetTensorShape(qData);
            auto kShape = GetTensorShape(kData);
            auto sinShape = GetTensorShape(sinData);

            int B = qShape[0];
            int qNumHeads = qShape[1];
            int kNumHeads = kShape[1];
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

                ropeApplyFloat(qPtr, qOutPtr, sinPtr, cosPtr, B, qNumHeads, N, headDim, sinB, sinH, hw, param->useFp16, param->interleaved);
                ropeApplyFloat(kPtr, kOutPtr, sinPtr, cosPtr, B, kNumHeads, N, headDim, sinB, sinH, hw, param->useFp16, param->interleaved);

            } else if (GetTensorType(qData) == TFCAPI_UINT8 && GetTensorType(kData) == TFCAPI_UINT8) {
                // Uint8 path: dequantize -> float RoPE -> requantize
                int qTotalElements = B * qNumHeads * N * headDim;
                int kTotalElements = B * kNumHeads * N * headDim;
                auto qQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
                auto kQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[1]);

                // Dequantize q and k to float
                float *qFloat = new float[qTotalElements];
                float *kFloat = new float[kTotalElements];
                DeQuantizeTensorData(qFloat, (uint8_t *) GetTensordata(qData), qTotalElements, qQuant);
                DeQuantizeTensorData(kFloat, (uint8_t *) GetTensordata(kData), kTotalElements, kQuant);

                // Apply RoPE in float
                float *qOutFloat = new float[qTotalElements];
                float *kOutFloat = new float[kTotalElements];
                ropeApplyFloat(qFloat, qOutFloat, sinPtr, cosPtr, B, qNumHeads, N, headDim, sinB, sinH, hw, param->useFp16, param->interleaved);
                ropeApplyFloat(kFloat, kOutFloat, sinPtr, cosPtr, B, kNumHeads, N, headDim, sinB, sinH, hw, param->useFp16, param->interleaved);

                // Requantize back to uint8
                auto qOutQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);
                auto kOutQuant = GetTensorQuantizeInfo(tfContext, info.OutputNames[1]);
                QuantizeTensorData((uint8_t *) GetTensordata(qOutData), qOutFloat, qTotalElements, qOutQuant);
                QuantizeTensorData((uint8_t *) GetTensordata(kOutData), kOutFloat, kTotalElements, kOutQuant);

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

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
#include <stdexcept>

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

        // 小工作量直接串行执行，避免 OpenMP 线程唤醒成本高于计算本身。
        static constexpr long long kMinParallelElements = 16 * 1024;

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
            TFCHECK_EQ(headDim % 2, 0);
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
        static void ropeApplyScalarPair(
                const float *q, const float *k,
                float *qOut, float *kOut,
                const float *sin, const float *cos,
                int B, int qNumHeads, int kNumHeads, int N, int headDim,
                int sinB, int sinH, int hw, bool interleaved = false) {
            int half = headDim / 2;
            int prefix = N - hw;
            const int tokensPerWork = 16;
            int tokenChunks = (N + tokensPerWork - 1) / tokensPerWork;
            int qItems = B * qNumHeads * tokenChunks;
            int kItems = B * kNumHeads * tokenChunks;
            int totalItems = qItems + kItems;
            long long totalElements =
                (long long)B * (qNumHeads + kNumHeads) * N * headDim;
            bool shouldParallel = false;
#ifdef _OPENMP
            shouldParallel =
                totalElements >= kMinParallelElements
                && omp_get_max_threads() > 1;
#endif

            // 将 Q/K 的所有 [b, head, token] 合并到同一个并行循环。
            // 相比先 Q 后 K 各启动一次线程组，GQA 的少量 KV heads 也能与 Q
            // 一起参与负载均衡。
            auto applyRange = [=](
                    const float *x, float *out, int numHeads,
                    int headRow, int nBegin, int nEnd) {
                int b = headRow / numHeads;
                int h = headRow % numHeads;
                int sb = (sinB == 1) ? 0 : b;
                int sh = (sinH == 1) ? 0 : h;
                long long headOffset = (long long)headRow * N * headDim;

                for (int n = nBegin; n < nEnd; n++) {
                    const float *xPtr = x + headOffset + (long long)n * headDim;
                    float *outPtr = out + headOffset + (long long)n * headDim;

                    if (n < prefix) {
                        memcpy(outPtr, xPtr, headDim * sizeof(float));
                        continue;
                    }

                    int sn = n - prefix;
                    const float *sinPtr =
                        sin + ((sb * sinH + sh) * hw + sn) * headDim;
                    const float *cosPtr =
                        cos + ((sb * sinH + sh) * hw + sn) * headDim;

                    if (interleaved) {
                        for (int d = 0; d < headDim; d += 2) {
                            outPtr[d] =
                                xPtr[d] * cosPtr[d]
                                - xPtr[d + 1] * sinPtr[d];
                            outPtr[d + 1] =
                                xPtr[d + 1] * cosPtr[d + 1]
                                + xPtr[d] * sinPtr[d + 1];
                        }
                    } else {
                        for (int d = 0; d < half; d++) {
                            outPtr[d] =
                                xPtr[d] * cosPtr[d]
                                - xPtr[d + half] * sinPtr[d];
                            outPtr[d + half] =
                                xPtr[d + half] * cosPtr[d + half]
                                + xPtr[d] * sinPtr[d + half];
                        }
                    }
                }
            };

            if (!shouldParallel) {
                for (int headRow = 0; headRow < B * qNumHeads; headRow++) {
                    applyRange(q, qOut, qNumHeads, headRow, 0, N);
                }
                for (int headRow = 0; headRow < B * kNumHeads; headRow++) {
                    applyRange(k, kOut, kNumHeads, headRow, 0, N);
                }
                return;
            }

            #pragma omp parallel for schedule(static)
            for (int item = 0; item < totalItems; item++) {
                bool isQ = item < qItems;
                int localItem = isQ ? item : item - qItems;
                int numHeads = isQ ? qNumHeads : kNumHeads;
                int tokenChunk = localItem % tokenChunks;
                int headRow = localItem / tokenChunks;
                int nBegin = tokenChunk * tokensPerWork;
                int nEnd = nBegin + tokensPerWork;
                if (nEnd > N) nEnd = N;
                applyRange(
                    isQ ? q : k,
                    isQ ? qOut : kOut,
                    numHeads,
                    headRow,
                    nBegin,
                    nEnd);
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
        static void ropeApplyNeonPair(
                const float *q, const float *k,
                float *qOut, float *kOut,
                const float *sin, const float *cos,
                int B, int qNumHeads, int kNumHeads, int N, int headDim,
                int sinB, int sinH, int hw) {
            int half = headDim / 2;
            int prefix = N - hw;
            const int tokensPerWork = 16;
            int tokenChunks = (N + tokensPerWork - 1) / tokensPerWork;
            int qItems = B * qNumHeads * tokenChunks;
            int kItems = B * kNumHeads * tokenChunks;
            int totalItems = qItems + kItems;
            long long totalElements =
                (long long)B * (qNumHeads + kNumHeads) * N * headDim;
            bool shouldParallel = false;
#ifdef _OPENMP
            shouldParallel =
                totalElements >= kMinParallelElements
                && omp_get_max_threads() > 1;
#endif

            auto applyRange = [=](
                    const float *x, float *out, int numHeads,
                    int headRow, int nBegin, int nEnd) {
                int b = headRow / numHeads;
                int h = headRow % numHeads;
                int sb = (sinB == 1) ? 0 : b;
                int sh = (sinH == 1) ? 0 : h;
                long long headOffset = (long long)headRow * N * headDim;

                for (int n = nBegin; n < nEnd; n++) {
                    const float *xPtr = x + headOffset + (long long)n * headDim;
                    float *outPtr = out + headOffset + (long long)n * headDim;

                    if (n < prefix) {
                        memcpy(outPtr, xPtr, headDim * sizeof(float));
                        continue;
                    }

                    int sn = n - prefix;
                    int scOffset = ((sb * sinH + sh) * hw + sn) * headDim;
                    const float *xFront = xPtr;
                    const float *xBack = xPtr + half;
                    const float *sinF = sin + scOffset;
                    const float *sinBk = sinF + half;
                    const float *cosF = cos + scOffset;
                    const float *cosBk = cosF + half;
                    float *outFront = outPtr;
                    float *outBack = outPtr + half;

                    int d = 0;
                    for (; d + 4 <= half; d += 4) {
                        float32x4_t vxF = vld1q_f32(xFront + d);
                        float32x4_t vxB = vld1q_f32(xBack + d);
                        float32x4_t vcosF = vld1q_f32(cosF + d);
                        float32x4_t vcosB = vld1q_f32(cosBk + d);
                        float32x4_t vsinF = vld1q_f32(sinF + d);
                        float32x4_t vsinB = vld1q_f32(sinBk + d);

                        float32x4_t vOutF = vmulq_f32(vxF, vcosF);
                        vOutF = vmlsq_f32(vOutF, vxB, vsinF);
                        vst1q_f32(outFront + d, vOutF);

                        float32x4_t vOutB = vmulq_f32(vxB, vcosB);
                        vOutB = vmlaq_f32(vOutB, vxF, vsinB);
                        vst1q_f32(outBack + d, vOutB);
                    }

                    for (; d < half; d++) {
                        outFront[d] =
                            xFront[d] * cosF[d] - xBack[d] * sinF[d];
                        outBack[d] =
                            xBack[d] * cosBk[d] + xFront[d] * sinBk[d];
                    }
                }
            };

            // 串行/小张量走按 head 的紧凑循环，避免分块调度开销。
            if (!shouldParallel) {
                for (int headRow = 0; headRow < B * qNumHeads; headRow++) {
                    applyRange(q, qOut, qNumHeads, headRow, 0, N);
                }
                for (int headRow = 0; headRow < B * kNumHeads; headRow++) {
                    applyRange(k, kOut, kNumHeads, headRow, 0, N);
                }
                return;
            }

            #pragma omp parallel for schedule(static)
            for (int item = 0; item < totalItems; item++) {
                bool isQ = item < qItems;
                int localItem = isQ ? item : item - qItems;
                int numHeads = isQ ? qNumHeads : kNumHeads;
                int tokenChunk = localItem % tokenChunks;
                int headRow = localItem / tokenChunks;
                int nBegin = tokenChunk * tokensPerWork;
                int nEnd = nBegin + tokensPerWork;
                if (nEnd > N) nEnd = N;
                applyRange(
                    isQ ? q : k,
                    isQ ? qOut : kOut,
                    numHeads,
                    headRow,
                    nBegin,
                    nEnd);
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
        static void ropeApplyFp16NeonPair(
                const float *q, const float *k,
                float *qOut, float *kOut,
                const float *sin, const float *cos,
                int B, int qNumHeads, int kNumHeads, int N, int headDim,
                int sinB, int sinH, int hw) {
            int half = headDim / 2;
            int prefix = N - hw;
            int tokensPerWork = N;
#ifdef _OPENMP
            if (omp_get_max_threads() > 1) tokensPerWork = 16;
#endif
            int tokenChunks = (N + tokensPerWork - 1) / tokensPerWork;
            int qItems = B * qNumHeads * tokenChunks;
            int kItems = B * kNumHeads * tokenChunks;
            int totalItems = qItems + kItems;
            long long totalElements =
                (long long)B * (qNumHeads + kNumHeads) * N * headDim;

            // Q/K 共用 sin/cos，只做一次 FP16 转换。
            int scTotal = sinB * sinH * hw * headDim;
            __fp16 *sinFp16 = new __fp16[scTotal];
            __fp16 *cosFp16 = new __fp16[scTotal];
            int vectorizedScTotal = (scTotal / 8) * 8;
            for (int i = vectorizedScTotal; i < scTotal; i++) {
                sinFp16[i] = (__fp16)sin[i];
                cosFp16[i] = (__fp16)cos[i];
            }

            // 一个线程组先并行转换 sin/cos，再处理 Q+K，避免重复创建线程组。
            #pragma omp parallel if(totalElements >= kMinParallelElements)
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < vectorizedScTotal; i += 8) {
                    float32x4_t sLo = vld1q_f32(sin + i);
                    float32x4_t sHi = vld1q_f32(sin + i + 4);
                    vst1q_f16(sinFp16 + i,
                              vcombine_f16(vcvt_f16_f32(sLo), vcvt_f16_f32(sHi)));
                    float32x4_t cLo = vld1q_f32(cos + i);
                    float32x4_t cHi = vld1q_f32(cos + i + 4);
                    vst1q_f16(cosFp16 + i,
                              vcombine_f16(vcvt_f16_f32(cLo), vcvt_f16_f32(cHi)));
                }

                #pragma omp for schedule(static)
                for (int item = 0; item < totalItems; item++) {
                    bool isQ = item < qItems;
                    int localItem = isQ ? item : item - qItems;
                    int numHeads = isQ ? qNumHeads : kNumHeads;
                    const float *x = isQ ? q : k;
                    float *out = isQ ? qOut : kOut;

                    int tokenChunk = localItem % tokenChunks;
                    int headRow = localItem / tokenChunks;
                    int b = headRow / numHeads;
                    int h = headRow % numHeads;
                    int sb = (sinB == 1) ? 0 : b;
                    int sh = (sinH == 1) ? 0 : h;
                    int nBegin = tokenChunk * tokensPerWork;
                    int nEnd = nBegin + tokensPerWork;
                    if (nEnd > N) nEnd = N;
                    long long headOffset = (long long)headRow * N * headDim;

                    for (int n = nBegin; n < nEnd; n++) {
                        const float *xPtr =
                            x + headOffset + (long long)n * headDim;
                        float *outPtr =
                            out + headOffset + (long long)n * headDim;

                        if (n < prefix) {
                            memcpy(outPtr, xPtr, headDim * sizeof(float));
                            continue;
                        }

                        int sn = n - prefix;
                        int scOffset =
                            ((sb * sinH + sh) * hw + sn) * headDim;
                        const float *xFront = xPtr;
                        const float *xBack = xPtr + half;
                        const __fp16 *sinF = sinFp16 + scOffset;
                        const __fp16 *sinBk = sinF + half;
                        const __fp16 *cosF = cosFp16 + scOffset;
                        const __fp16 *cosBk = cosF + half;
                        float *outFront = outPtr;
                        float *outBack = outPtr + half;

                        const float *sinF32 = sin + scOffset;
                        const float *sinBk32 = sinF32 + half;
                        const float *cosF32 = cos + scOffset;
                        const float *cosBk32 = cosF32 + half;

                        int d = 0;
                        for (; d + 8 <= half; d += 8) {
                            float16x8_t vxF = vcombine_f16(
                                vcvt_f16_f32(vld1q_f32(xFront + d)),
                                vcvt_f16_f32(vld1q_f32(xFront + d + 4)));
                            float16x8_t vxB = vcombine_f16(
                                vcvt_f16_f32(vld1q_f32(xBack + d)),
                                vcvt_f16_f32(vld1q_f32(xBack + d + 4)));
                            float16x8_t vsinF = vld1q_f16(sinF + d);
                            float16x8_t vsinB = vld1q_f16(sinBk + d);
                            float16x8_t vcosF = vld1q_f16(cosF + d);
                            float16x8_t vcosB = vld1q_f16(cosBk + d);

                            float16x8_t vOutF = vmulq_f16(vxF, vcosF);
                            vOutF = vfmsq_f16(vOutF, vxB, vsinF);
                            float16x8_t vOutB = vmulq_f16(vxB, vcosB);
                            vOutB = vfmaq_f16(vOutB, vxF, vsinB);

                            vst1q_f32(
                                outFront + d,
                                vcvt_f32_f16(vget_low_f16(vOutF)));
                            vst1q_f32(
                                outFront + d + 4,
                                vcvt_f32_f16(vget_high_f16(vOutF)));
                            vst1q_f32(
                                outBack + d,
                                vcvt_f32_f16(vget_low_f16(vOutB)));
                            vst1q_f32(
                                outBack + d + 4,
                                vcvt_f32_f16(vget_high_f16(vOutB)));
                        }

                        for (; d + 4 <= half; d += 4) {
                            float32x4_t vxF = vld1q_f32(xFront + d);
                            float32x4_t vxB = vld1q_f32(xBack + d);
                            float32x4_t vOutF =
                                vmulq_f32(vxF, vld1q_f32(cosF32 + d));
                            vOutF = vmlsq_f32(
                                vOutF, vxB, vld1q_f32(sinF32 + d));
                            vst1q_f32(outFront + d, vOutF);

                            float32x4_t vOutB =
                                vmulq_f32(vxB, vld1q_f32(cosBk32 + d));
                            vOutB = vmlaq_f32(
                                vOutB, vxF, vld1q_f32(sinBk32 + d));
                            vst1q_f32(outBack + d, vOutB);
                        }

                        for (; d < half; d++) {
                            outFront[d] =
                                xFront[d] * cosF32[d]
                                - xBack[d] * sinF32[d];
                            outBack[d] =
                                xBack[d] * cosBk32[d]
                                + xFront[d] * sinBk32[d];
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
        static void ropeApplyFloatPair(
                const float *q, const float *k,
                float *qOut, float *kOut,
                const float *sin, const float *cos,
                int B, int qNumHeads, int kNumHeads, int N, int headDim,
                int sinB, int sinH, int hw, bool useFp16, bool interleaved = false) {
            if (interleaved) {
                ropeApplyScalarPair(
                    q, k, qOut, kOut, sin, cos,
                    B, qNumHeads, kNumHeads, N, headDim,
                    sinB, sinH, hw, true);
                return;
            }
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            if (useFp16) {
                ropeApplyFp16NeonPair(
                    q, k, qOut, kOut, sin, cos,
                    B, qNumHeads, kNumHeads, N, headDim,
                    sinB, sinH, hw);
            } else {
                ropeApplyNeonPair(
                    q, k, qOut, kOut, sin, cos,
                    B, qNumHeads, kNumHeads, N, headDim,
                    sinB, sinH, hw);
            }
#elif defined(__aarch64__)
            (void)useFp16;
            ropeApplyNeonPair(
                q, k, qOut, kOut, sin, cos,
                B, qNumHeads, kNumHeads, N, headDim,
                sinB, sinH, hw);
#else
            (void)useFp16;
            ropeApplyScalarPair(
                q, k, qOut, kOut, sin, cos,
                B, qNumHeads, kNumHeads, N, headDim,
                sinB, sinH, hw);
#endif
        }

        struct Uint8RopeQuant {
            int inputZeroPoint;
            int outputZeroPoint;
            float inputToOutputScale;
        };

        static Uint8RopeQuant makeUint8RopeQuant(
                Quantization inputQuant, Quantization outputQuant,
                const string &inputName, const string &outputName) {
            if (!inputQuant.IsValid() || !outputQuant.IsValid()) {
                throw std::runtime_error(
                    "ApplyRope UINT8 requires valid quantization for "
                    + inputName + " and " + outputName);
            }

            const auto &inputScales = GetQuantizationScale(inputQuant);
            const auto &inputZeros = GetQuantizationZeroPoint(inputQuant);
            const auto &outputScales = GetQuantizationScale(outputQuant);
            const auto &outputZeros = GetQuantizationZeroPoint(outputQuant);
            if (inputScales.size() != 1 || inputZeros.size() != 1
                    || outputScales.size() != 1 || outputZeros.size() != 1) {
                throw std::runtime_error(
                    "ApplyRope fused UINT8 only supports per-tensor "
                    "activation quantization: input=" + inputName
                    + ", output=" + outputName);
            }
            if (!(inputScales[0] > 0.f) || !(outputScales[0] > 0.f)
                    || !std::isfinite(inputScales[0])
                    || !std::isfinite(outputScales[0])) {
                throw std::runtime_error(
                    "ApplyRope UINT8 got invalid quantization scale: input="
                    + inputName + ", output=" + outputName);
            }
            if (inputZeros[0] < 0 || inputZeros[0] > 255
                    || outputZeros[0] < 0 || outputZeros[0] > 255) {
                throw std::runtime_error(
                    "ApplyRope UINT8 got zero-point outside [0, 255]: input="
                    + inputName + ", output=" + outputName);
            }

            return {
                inputZeros[0],
                outputZeros[0],
                inputScales[0] / outputScales[0]
            };
        }

        static inline uint8_t quantizeCenteredScalar(
                float centeredValue, const Uint8RopeQuant &quant) {
            float encoded = std::nearbyint(
                centeredValue * quant.inputToOutputScale
                + (float)quant.outputZeroPoint);
            if (encoded <= 0.f) return 0;
            if (encoded >= 255.f) return 255;
            return (uint8_t)encoded;
        }

        static void ropeApplyUint8ScalarRange(
                const uint8_t *x, uint8_t *out,
                const float *sin, const float *cos,
                int numHeads, int N, int headDim,
                int sinB, int sinH, int hw,
                int headRow, int nBegin, int nEnd,
                const Uint8RopeQuant &quant, bool interleaved) {
            int half = headDim / 2;
            int prefix = N - hw;
            int b = headRow / numHeads;
            int h = headRow % numHeads;
            int sb = (sinB == 1) ? 0 : b;
            int sh = (sinH == 1) ? 0 : h;
            long long headOffset = (long long)headRow * N * headDim;

            for (int n = nBegin; n < nEnd; n++) {
                const uint8_t *xPtr =
                    x + headOffset + (long long)n * headDim;
                uint8_t *outPtr =
                    out + headOffset + (long long)n * headDim;

                if (n < prefix) {
                    for (int d = 0; d < headDim; d++) {
                        float centered =
                            (float)((int)xPtr[d] - quant.inputZeroPoint);
                        outPtr[d] = quantizeCenteredScalar(centered, quant);
                    }
                    continue;
                }

                int sn = n - prefix;
                int scOffset = ((sb * sinH + sh) * hw + sn) * headDim;
                const float *sinPtr = sin + scOffset;
                const float *cosPtr = cos + scOffset;

                if (interleaved) {
                    for (int d = 0; d < headDim; d += 2) {
                        float x0 =
                            (float)((int)xPtr[d] - quant.inputZeroPoint);
                        float x1 =
                            (float)((int)xPtr[d + 1] - quant.inputZeroPoint);
                        outPtr[d] = quantizeCenteredScalar(
                            x0 * cosPtr[d] - x1 * sinPtr[d], quant);
                        outPtr[d + 1] = quantizeCenteredScalar(
                            x1 * cosPtr[d + 1]
                            + x0 * sinPtr[d + 1],
                            quant);
                    }
                } else {
                    for (int d = 0; d < half; d++) {
                        float xFront =
                            (float)((int)xPtr[d] - quant.inputZeroPoint);
                        float xBack =
                            (float)((int)xPtr[d + half]
                                    - quant.inputZeroPoint);
                        outPtr[d] = quantizeCenteredScalar(
                            xFront * cosPtr[d]
                            - xBack * sinPtr[d],
                            quant);
                        outPtr[d + half] = quantizeCenteredScalar(
                            xBack * cosPtr[d + half]
                            + xFront * sinPtr[d + half],
                            quant);
                    }
                }
            }
        }

#ifdef __aarch64__
        static inline void loadCenteredUint8x8(
                const uint8_t *src, int zeroPoint,
                float32x4_t &low, float32x4_t &high) {
            uint16x8_t wide = vmovl_u8(vld1_u8(src));
            int16x8_t centered = vsubq_s16(
                vreinterpretq_s16_u16(wide),
                vdupq_n_s16((int16_t)zeroPoint));
            low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(centered)));
            high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(centered)));
        }

        static inline uint8x8_t quantizeCenteredFloat32x8(
                float32x4_t low, float32x4_t high,
                const Uint8RopeQuant &quant) {
            float32x4_t outputZero =
                vdupq_n_f32((float)quant.outputZeroPoint);
            int32x4_t lowInt = vcvtnq_s32_f32(
                vmlaq_n_f32(
                    outputZero, low, quant.inputToOutputScale));
            int32x4_t highInt = vcvtnq_s32_f32(
                vmlaq_n_f32(
                    outputZero, high, quant.inputToOutputScale));
            uint16x8_t narrowed16 = vcombine_u16(
                vqmovun_s32(lowInt), vqmovun_s32(highInt));
            return vqmovn_u16(narrowed16);
        }

        static inline void requantizeUint8Neon(
                const uint8_t *src, uint8_t *dst, int count,
                const Uint8RopeQuant &quant) {
            int d = 0;
            for (; d + 8 <= count; d += 8) {
                float32x4_t low;
                float32x4_t high;
                loadCenteredUint8x8(
                    src + d, quant.inputZeroPoint, low, high);
                vst1_u8(
                    dst + d,
                    quantizeCenteredFloat32x8(low, high, quant));
            }
            for (; d < count; d++) {
                float centered =
                    (float)((int)src[d] - quant.inputZeroPoint);
                dst[d] = quantizeCenteredScalar(centered, quant);
            }
        }

        static void ropeApplyUint8NeonRange(
                const uint8_t *x, uint8_t *out,
                const float *sin, const float *cos,
                int numHeads, int N, int headDim,
                int sinB, int sinH, int hw,
                int headRow, int nBegin, int nEnd,
                const Uint8RopeQuant &quant, bool interleaved) {
            int half = headDim / 2;
            int prefix = N - hw;
            int b = headRow / numHeads;
            int h = headRow % numHeads;
            int sb = (sinB == 1) ? 0 : b;
            int sh = (sinH == 1) ? 0 : h;
            long long headOffset = (long long)headRow * N * headDim;

            for (int n = nBegin; n < nEnd; n++) {
                const uint8_t *xPtr =
                    x + headOffset + (long long)n * headDim;
                uint8_t *outPtr =
                    out + headOffset + (long long)n * headDim;

                if (n < prefix) {
                    requantizeUint8Neon(
                        xPtr, outPtr, headDim, quant);
                    continue;
                }

                int sn = n - prefix;
                int scOffset = ((sb * sinH + sh) * hw + sn) * headDim;
                const float *sinPtr = sin + scOffset;
                const float *cosPtr = cos + scOffset;

                if (interleaved) {
                    int d = 0;
                    for (; d + 16 <= headDim; d += 16) {
                        uint8x8x2_t inputPair = vld2_u8(xPtr + d);
                        float32x4_t xEvenLow;
                        float32x4_t xEvenHigh;
                        float32x4_t xOddLow;
                        float32x4_t xOddHigh;
                        uint16x8_t evenWide = vmovl_u8(inputPair.val[0]);
                        uint16x8_t oddWide = vmovl_u8(inputPair.val[1]);
                        int16x8_t evenCentered = vsubq_s16(
                            vreinterpretq_s16_u16(evenWide),
                            vdupq_n_s16(
                                (int16_t)quant.inputZeroPoint));
                        int16x8_t oddCentered = vsubq_s16(
                            vreinterpretq_s16_u16(oddWide),
                            vdupq_n_s16(
                                (int16_t)quant.inputZeroPoint));
                        xEvenLow = vcvtq_f32_s32(
                            vmovl_s16(vget_low_s16(evenCentered)));
                        xEvenHigh = vcvtq_f32_s32(
                            vmovl_s16(vget_high_s16(evenCentered)));
                        xOddLow = vcvtq_f32_s32(
                            vmovl_s16(vget_low_s16(oddCentered)));
                        xOddHigh = vcvtq_f32_s32(
                            vmovl_s16(vget_high_s16(oddCentered)));

                        float32x4x2_t sinLow = vld2q_f32(sinPtr + d);
                        float32x4x2_t sinHigh =
                            vld2q_f32(sinPtr + d + 8);
                        float32x4x2_t cosLow = vld2q_f32(cosPtr + d);
                        float32x4x2_t cosHigh =
                            vld2q_f32(cosPtr + d + 8);

                        float32x4_t outEvenLow =
                            vmulq_f32(xEvenLow, cosLow.val[0]);
                        outEvenLow = vmlsq_f32(
                            outEvenLow, xOddLow, sinLow.val[0]);
                        float32x4_t outEvenHigh =
                            vmulq_f32(xEvenHigh, cosHigh.val[0]);
                        outEvenHigh = vmlsq_f32(
                            outEvenHigh, xOddHigh, sinHigh.val[0]);

                        float32x4_t outOddLow =
                            vmulq_f32(xOddLow, cosLow.val[1]);
                        outOddLow = vmlaq_f32(
                            outOddLow, xEvenLow, sinLow.val[1]);
                        float32x4_t outOddHigh =
                            vmulq_f32(xOddHigh, cosHigh.val[1]);
                        outOddHigh = vmlaq_f32(
                            outOddHigh, xEvenHigh, sinHigh.val[1]);

                        uint8x8x2_t outputPair;
                        outputPair.val[0] =
                            quantizeCenteredFloat32x8(
                                outEvenLow, outEvenHigh, quant);
                        outputPair.val[1] =
                            quantizeCenteredFloat32x8(
                                outOddLow, outOddHigh, quant);
                        vst2_u8(outPtr + d, outputPair);
                    }

                    for (; d < headDim; d += 2) {
                        float x0 =
                            (float)((int)xPtr[d] - quant.inputZeroPoint);
                        float x1 =
                            (float)((int)xPtr[d + 1]
                                    - quant.inputZeroPoint);
                        outPtr[d] = quantizeCenteredScalar(
                            x0 * cosPtr[d] - x1 * sinPtr[d], quant);
                        outPtr[d + 1] = quantizeCenteredScalar(
                            x1 * cosPtr[d + 1]
                            + x0 * sinPtr[d + 1],
                            quant);
                    }
                } else {
                    int d = 0;
                    for (; d + 8 <= half; d += 8) {
                        float32x4_t xFrontLow;
                        float32x4_t xFrontHigh;
                        float32x4_t xBackLow;
                        float32x4_t xBackHigh;
                        loadCenteredUint8x8(
                            xPtr + d,
                            quant.inputZeroPoint,
                            xFrontLow,
                            xFrontHigh);
                        loadCenteredUint8x8(
                            xPtr + half + d,
                            quant.inputZeroPoint,
                            xBackLow,
                            xBackHigh);

                        float32x4_t outFrontLow =
                            vmulq_f32(
                                xFrontLow, vld1q_f32(cosPtr + d));
                        outFrontLow = vmlsq_f32(
                            outFrontLow,
                            xBackLow,
                            vld1q_f32(sinPtr + d));
                        float32x4_t outFrontHigh =
                            vmulq_f32(
                                xFrontHigh,
                                vld1q_f32(cosPtr + d + 4));
                        outFrontHigh = vmlsq_f32(
                            outFrontHigh,
                            xBackHigh,
                            vld1q_f32(sinPtr + d + 4));

                        float32x4_t outBackLow =
                            vmulq_f32(
                                xBackLow,
                                vld1q_f32(cosPtr + half + d));
                        outBackLow = vmlaq_f32(
                            outBackLow,
                            xFrontLow,
                            vld1q_f32(sinPtr + half + d));
                        float32x4_t outBackHigh =
                            vmulq_f32(
                                xBackHigh,
                                vld1q_f32(cosPtr + half + d + 4));
                        outBackHigh = vmlaq_f32(
                            outBackHigh,
                            xFrontHigh,
                            vld1q_f32(sinPtr + half + d + 4));

                        vst1_u8(
                            outPtr + d,
                            quantizeCenteredFloat32x8(
                                outFrontLow, outFrontHigh, quant));
                        vst1_u8(
                            outPtr + half + d,
                            quantizeCenteredFloat32x8(
                                outBackLow, outBackHigh, quant));
                    }

                    for (; d < half; d++) {
                        float xFront =
                            (float)((int)xPtr[d] - quant.inputZeroPoint);
                        float xBack =
                            (float)((int)xPtr[d + half]
                                    - quant.inputZeroPoint);
                        outPtr[d] = quantizeCenteredScalar(
                            xFront * cosPtr[d]
                            - xBack * sinPtr[d],
                            quant);
                        outPtr[d + half] = quantizeCenteredScalar(
                            xBack * cosPtr[d + half]
                            + xFront * sinPtr[d + half],
                            quant);
                    }
                }
            }
        }
#endif

        static void ropeApplyUint8Pair(
                const uint8_t *q, const uint8_t *k,
                uint8_t *qOut, uint8_t *kOut,
                const float *sin, const float *cos,
                int B, int qNumHeads, int kNumHeads, int N, int headDim,
                int sinB, int sinH, int hw,
                const Uint8RopeQuant &qQuant,
                const Uint8RopeQuant &kQuant,
                bool interleaved) {
            const int tokensPerWork = 16;
            int tokenChunks = (N + tokensPerWork - 1) / tokensPerWork;
            int qItems = B * qNumHeads * tokenChunks;
            int kItems = B * kNumHeads * tokenChunks;
            int totalItems = qItems + kItems;
            long long totalElements =
                (long long)B * (qNumHeads + kNumHeads) * N * headDim;
            bool shouldParallel = false;
#ifdef _OPENMP
            shouldParallel =
                totalElements >= kMinParallelElements
                && omp_get_max_threads() > 1;
#endif

            auto applyRange = [=](
                    const uint8_t *x, uint8_t *out, int numHeads,
                    int headRow, int nBegin, int nEnd,
                    const Uint8RopeQuant &quant) {
#ifdef __aarch64__
                ropeApplyUint8NeonRange(
                    x, out, sin, cos,
                    numHeads, N, headDim, sinB, sinH, hw,
                    headRow, nBegin, nEnd, quant, interleaved);
#else
                ropeApplyUint8ScalarRange(
                    x, out, sin, cos,
                    numHeads, N, headDim, sinB, sinH, hw,
                    headRow, nBegin, nEnd, quant, interleaved);
#endif
            };

            if (!shouldParallel) {
                for (int headRow = 0; headRow < B * qNumHeads; headRow++) {
                    applyRange(
                        q, qOut, qNumHeads, headRow, 0, N, qQuant);
                }
                for (int headRow = 0; headRow < B * kNumHeads; headRow++) {
                    applyRange(
                        k, kOut, kNumHeads, headRow, 0, N, kQuant);
                }
                return;
            }

            #pragma omp parallel for schedule(static)
            for (int item = 0; item < totalItems; item++) {
                bool isQ = item < qItems;
                int localItem = isQ ? item : item - qItems;
                int numHeads = isQ ? qNumHeads : kNumHeads;
                int tokenChunk = localItem % tokenChunks;
                int headRow = localItem / tokenChunks;
                int nBegin = tokenChunk * tokensPerWork;
                int nEnd = nBegin + tokensPerWork;
                if (nEnd > N) nEnd = N;
                applyRange(
                    isQ ? q : k,
                    isQ ? qOut : kOut,
                    numHeads,
                    headRow,
                    nBegin,
                    nEnd,
                    isQ ? qQuant : kQuant);
            }
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

                ropeApplyFloatPair(
                    qPtr, kPtr, qOutPtr, kOutPtr, sinPtr, cosPtr,
                    B, qNumHeads, kNumHeads, N, headDim,
                    sinB, sinH, hw, param->useFp16, param->interleaved);

            } else if (GetTensorType(qData) == TFCAPI_UINT8 && GetTensorType(kData) == TFCAPI_UINT8) {
                auto qQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
                auto kQuant = GetTensorQuantizeInfo(tfContext, info.InputNames[1]);
                auto qOutQuant =
                    GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);
                auto kOutQuant =
                    GetTensorQuantizeInfo(tfContext, info.OutputNames[1]);
                auto qRopeQuant = makeUint8RopeQuant(
                    qQuant, qOutQuant,
                    info.InputNames[0], info.OutputNames[0]);
                auto kRopeQuant = makeUint8RopeQuant(
                    kQuant, kOutQuant,
                    info.InputNames[1], info.OutputNames[1]);

                auto qUint8 = (uint8_t *)GetTensordata(qData);
                auto kUint8 = (uint8_t *)GetTensordata(kData);
                auto qOutUint8 = (uint8_t *)GetTensordata(qOutData);
                auto kOutUint8 = (uint8_t *)GetTensordata(kOutData);

                ropeApplyUint8Pair(
                    qUint8, kUint8, qOutUint8, kOutUint8,
                    sinPtr, cosPtr,
                    B, qNumHeads, kNumHeads, N, headDim,
                    sinB, sinH, hw,
                    qRopeQuant, kRopeQuant, param->interleaved);
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

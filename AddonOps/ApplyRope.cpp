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
#include <cstdint>
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
        enum class RopeLayout {
            BHND,
            BHDN,
        };

        struct ApplyRopeParam {
            bool useFp16 = false;  // 使用 FP16 NEON 加速 (精度略降, 速度更快)
            bool interleaved = false;  // Adjacent complex pairs, used by MoonViT.
            RopeLayout inputLayout = RopeLayout::BHND;
            RopeLayout qOutputLayout = RopeLayout::BHND;
            RopeLayout kOutputLayout = RopeLayout::BHND;
        };

        static RopeLayout parseLayout(
                const string &layout, RopeLayout fallback) {
            if (layout.empty()) return fallback;
            if (layout == "BHND") return RopeLayout::BHND;
            if (layout == "BHDN") return RopeLayout::BHDN;
            throw std::runtime_error(
                "ApplyRope layout must be BHND or BHDN, got " + layout);
        }

        static void parseParamJson(
                const string &jsonText, ApplyRopeParam *paramOut) {
            string err;
            const json11::Json param = json11::Json::parse(jsonText, err);
            if (!err.empty()) return;
            paramOut->useFp16 = param["useFp16"].bool_value();
            paramOut->interleaved = param["interleaved"].bool_value();
            const RopeLayout common = parseLayout(
                param["layout"].string_value(), RopeLayout::BHND);
            paramOut->inputLayout = parseLayout(
                param["inputLayout"].string_value(), common);
            paramOut->qOutputLayout = parseLayout(
                param["qOutputLayout"].string_value(),
                paramOut->inputLayout);
            paramOut->kOutputLayout = parseLayout(
                param["kOutputLayout"].string_value(),
                paramOut->inputLayout);
        }

        // 小工作量直接串行执行，避免 OpenMP 线程唤醒成本高于计算本身。
        static constexpr long long kMinParallelElements = 16 * 1024;

        void Prepare(TFContext tfContext, TFNode node) {
            ApplyRopeParam *p = new ApplyRopeParam();
            parseParamJson(GetNodeCustomJsonStr(node), p);

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
            ApplyRopeParam reshapeParam;
            parseParamJson(GetNodeCustomJsonStr(node), &reshapeParam);
            auto param = &reshapeParam;
            const bool inputSequenceLast =
                param->inputLayout == RopeLayout::BHDN;
            const int tokenAxis = inputSequenceLast ? 3 : 2;
            const int dimAxis = inputSequenceLast ? 2 : 3;

            // q/k are [B,H,N,D] by default or [B,H,D,N] in Conv-native mode.
            // ViT uses equal head counts, while Qwen-style GQA uses more
            // query heads than KV heads.
            TFCHECK_EQ(qShape.size(), 4);
            TFCHECK_EQ(kShape.size(), 4);
            TFCHECK_EQ(qShape[0], kShape[0]);
            TFCHECK_EQ(qShape[tokenAxis], kShape[tokenAxis]);
            TFCHECK_EQ(qShape[dimAxis], kShape[dimAxis]);

            // Tables follow the selected layout as well: [...,hw,D] for
            // BHND and [...,D,hw] for BHDN.
            TFCHECK_EQ(sinShape.size(), cosShape.size());
            for (size_t i = 0; i < sinShape.size(); i++) {
                TFCHECK_EQ(sinShape[i], cosShape[i]);
            }

            // head_dim must match
            int headDim = qShape[dimAxis];
            TFCHECK_EQ(headDim % 2, 0);
            const int tableDimAxis = inputSequenceLast
                ? (int)sinShape.size() - 2
                : (int)sinShape.size() - 1;
            const int tableTokenAxis = inputSequenceLast
                ? (int)sinShape.size() - 1
                : (int)sinShape.size() - 2;
            TFCHECK_EQ(sinShape[tableDimAxis], headDim);
            TFCHECK_EQ(cosShape[tableDimAxis], headDim);

            // hw (sin's seq dimension) must be <= N (q's seq dimension)
            int N = qShape[tokenAxis];
            int hw = sinShape[tableTokenAxis];
            TFCHECK_GE(N, hw);

            auto outputShape = [](int batch, int heads, int tokens, int dim,
                                  RopeLayout layout) {
                return layout == RopeLayout::BHND
                    ? vector<int>{batch, heads, tokens, dim}
                    : vector<int>{batch, heads, dim, tokens};
            };
            ReSizeTensor(
                qOutData,
                outputShape(
                    qShape[0], qShape[1], N, headDim,
                    param->qOutputLayout));
            SetTensorType(qOutData, GetTensorType(qData));
            ReSizeTensor(
                kOutData,
                outputShape(
                    kShape[0], kShape[1], N, headDim,
                    param->kOutputLayout));
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

        // Native FP16 tensor path.  The older useFp16 implementation above
        // deliberately keeps FP32 input/output and only performs its inner
        // arithmetic in FP16.  Qwen-prefill has real FLOAT16 tensors, so it
        // needs a separate dispatcher and must never reinterpret their
        // storage as float pointers.
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        static void ropeApplyNativeFp16Pair(
                const uint16_t *qStorage, const uint16_t *kStorage,
                uint16_t *qOutStorage, uint16_t *kOutStorage,
                const float *sin, const float *cos,
                int B, int qNumHeads, int kNumHeads, int N, int headDim,
                int sinB, int sinH, int hw, bool interleaved) {
            const __fp16 *q = reinterpret_cast<const __fp16 *>(qStorage);
            const __fp16 *k = reinterpret_cast<const __fp16 *>(kStorage);
            __fp16 *qOut = reinterpret_cast<__fp16 *>(qOutStorage);
            __fp16 *kOut = reinterpret_cast<__fp16 *>(kOutStorage);
            const int half = headDim / 2;
            const int prefix = N - hw;
            const int tokensPerWork = 16;
            const int tokenChunks = (N + tokensPerWork - 1) / tokensPerWork;
            const int qItems = B * qNumHeads * tokenChunks;
            const int kItems = B * kNumHeads * tokenChunks;
            const int totalItems = qItems + kItems;
            const long long totalElements =
                (long long)B * (qNumHeads + kNumHeads) * N * headDim;

            // Q and K share the same RoPE tables.  Convert the tables once
            // instead of paying FP32->FP16 conversion in every head.
            const int tableElements = sinB * sinH * hw * headDim;
            __fp16 *sinFp16 = new __fp16[tableElements];
            __fp16 *cosFp16 = new __fp16[tableElements];
            const int vectorized = (tableElements / 8) * 8;
            for (int index = vectorized; index < tableElements; ++index) {
                sinFp16[index] = (__fp16)sin[index];
                cosFp16[index] = (__fp16)cos[index];
            }

            #pragma omp parallel if(totalElements >= kMinParallelElements)
            {
                #pragma omp for schedule(static)
                for (int index = 0; index < vectorized; index += 8) {
                    vst1q_f16(
                        sinFp16 + index,
                        vcombine_f16(
                            vcvt_f16_f32(vld1q_f32(sin + index)),
                            vcvt_f16_f32(vld1q_f32(sin + index + 4))));
                    vst1q_f16(
                        cosFp16 + index,
                        vcombine_f16(
                            vcvt_f16_f32(vld1q_f32(cos + index)),
                            vcvt_f16_f32(vld1q_f32(cos + index + 4))));
                }

                #pragma omp for schedule(static)
                for (int item = 0; item < totalItems; ++item) {
                    const bool isQ = item < qItems;
                    const int localItem = isQ ? item : item - qItems;
                    const int numHeads = isQ ? qNumHeads : kNumHeads;
                    const __fp16 *input = isQ ? q : k;
                    __fp16 *output = isQ ? qOut : kOut;
                    const int tokenChunk = localItem % tokenChunks;
                    const int headRow = localItem / tokenChunks;
                    const int batch = headRow / numHeads;
                    const int head = headRow % numHeads;
                    const int sinBatch = sinB == 1 ? 0 : batch;
                    const int sinHead = sinH == 1 ? 0 : head;
                    const int begin = tokenChunk * tokensPerWork;
                    const int end = std::min(N, begin + tokensPerWork);
                    const long long headOffset =
                        (long long)headRow * N * headDim;

                    for (int token = begin; token < end; ++token) {
                        const __fp16 *x = input + headOffset
                            + (long long)token * headDim;
                        __fp16 *y = output + headOffset
                            + (long long)token * headDim;
                        if (token < prefix) {
                            memcpy(y, x, headDim * sizeof(__fp16));
                            continue;
                        }
                        const int ropeToken = token - prefix;
                        const int tableOffset =
                            ((sinBatch * sinH + sinHead) * hw + ropeToken)
                            * headDim;
                        const __fp16 *sinRow = sinFp16 + tableOffset;
                        const __fp16 *cosRow = cosFp16 + tableOffset;

                        if (interleaved) {
                            for (int dim = 0; dim < headDim; dim += 2) {
                                const __fp16 x0 = x[dim];
                                const __fp16 x1 = x[dim + 1];
                                y[dim] = x0 * cosRow[dim]
                                    - x1 * sinRow[dim];
                                y[dim + 1] = x1 * cosRow[dim + 1]
                                    + x0 * sinRow[dim + 1];
                            }
                            continue;
                        }

                        int dim = 0;
                        for (; dim + 8 <= half; dim += 8) {
                            const float16x8_t front = vld1q_f16(x + dim);
                            const float16x8_t back = vld1q_f16(x + half + dim);
                            float16x8_t outFront = vmulq_f16(
                                front, vld1q_f16(cosRow + dim));
                            outFront = vfmsq_f16(
                                outFront, back, vld1q_f16(sinRow + dim));
                            float16x8_t outBack = vmulq_f16(
                                back, vld1q_f16(cosRow + half + dim));
                            outBack = vfmaq_f16(
                                outBack, front,
                                vld1q_f16(sinRow + half + dim));
                            vst1q_f16(y + dim, outFront);
                            vst1q_f16(y + half + dim, outBack);
                        }
                        for (; dim < half; ++dim) {
                            const __fp16 front = x[dim];
                            const __fp16 back = x[half + dim];
                            y[dim] = front * cosRow[dim]
                                - back * sinRow[dim];
                            y[half + dim] = back * cosRow[half + dim]
                                + front * sinRow[half + dim];
                        }
                    }
                }
            }
            delete[] sinFp16;
            delete[] cosFp16;
        }
#endif
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

        static inline long long ropeTensorOffset(
                RopeLayout layout, int headRow, int token, int dim,
                int N, int headDim) {
            if (layout == RopeLayout::BHND) {
                return ((long long)headRow * N + token) * headDim + dim;
            }
            return ((long long)headRow * headDim + dim) * N + token;
        }

        static inline long long ropeTableOffset(
                RopeLayout layout, int tableHead, int token, int dim,
                int hw, int headDim) {
            if (layout == RopeLayout::BHND) {
                return ((long long)tableHead * hw + token) * headDim + dim;
            }
            return ((long long)tableHead * headDim + dim) * hw + token;
        }

        // General layout path.  The established BHND->BHND fast kernels above
        // remain unchanged.  This path is used by Conv-native ViT, where the
        // input/table are BHDN and Q/K intentionally have different output
        // layouts so the following MatMuls need no trans flags.
        static void ropeApplyFloatPairLayouts(
                const float *q, const float *k,
                float *qOut, float *kOut,
                const float *sin, const float *cos,
                int B, int qNumHeads, int kNumHeads, int N, int headDim,
                int sinB, int sinH, int hw, bool interleaved,
                RopeLayout inputLayout,
                RopeLayout qOutputLayout,
                RopeLayout kOutputLayout) {
            const int half = headDim / 2;
            const int prefix = N - hw;
            const int qRows = B * qNumHeads;
            const int kRows = B * kNumHeads;
            const int totalRows = qRows + kRows;
            const long long totalElements =
                (long long)totalRows * N * headDim;
            bool shouldParallel = false;
#ifdef _OPENMP
            shouldParallel = totalElements >= kMinParallelElements
                && omp_get_max_threads() > 1;
#endif

            auto applyHead = [=](
                    const float *input, float *output, int numHeads,
                    int headRow, RopeLayout outputLayout) {
                const int batch = headRow / numHeads;
                const int head = headRow % numHeads;
                const int sinBatch = sinB == 1 ? 0 : batch;
                const int sinHead = sinH == 1 ? 0 : head;
                const int tableHead = sinBatch * sinH + sinHead;
                for (int token = 0; token < N; ++token) {
                    if (token < prefix) {
                        for (int dim = 0; dim < headDim; ++dim) {
                            output[ropeTensorOffset(
                                outputLayout, headRow, token, dim,
                                N, headDim)] = input[ropeTensorOffset(
                                inputLayout, headRow, token, dim,
                                N, headDim)];
                        }
                        continue;
                    }
                    const int ropeToken = token - prefix;
                    if (interleaved) {
                        for (int dim = 0; dim < headDim; dim += 2) {
                            const float x0 = input[ropeTensorOffset(
                                inputLayout, headRow, token, dim,
                                N, headDim)];
                            const float x1 = input[ropeTensorOffset(
                                inputLayout, headRow, token, dim + 1,
                                N, headDim)];
                            const long long table0 = ropeTableOffset(
                                inputLayout, tableHead, ropeToken, dim,
                                hw, headDim);
                            const long long table1 = ropeTableOffset(
                                inputLayout, tableHead, ropeToken, dim + 1,
                                hw, headDim);
                            output[ropeTensorOffset(
                                outputLayout, headRow, token, dim,
                                N, headDim)] =
                                x0 * cos[table0] - x1 * sin[table0];
                            output[ropeTensorOffset(
                                outputLayout, headRow, token, dim + 1,
                                N, headDim)] =
                                x1 * cos[table1] + x0 * sin[table1];
                        }
                    } else {
                        for (int dim = 0; dim < half; ++dim) {
                            const float front = input[ropeTensorOffset(
                                inputLayout, headRow, token, dim,
                                N, headDim)];
                            const float back = input[ropeTensorOffset(
                                inputLayout, headRow, token, dim + half,
                                N, headDim)];
                            const long long tableFront = ropeTableOffset(
                                inputLayout, tableHead, ropeToken, dim,
                                hw, headDim);
                            const long long tableBack = ropeTableOffset(
                                inputLayout, tableHead, ropeToken, dim + half,
                                hw, headDim);
                            output[ropeTensorOffset(
                                outputLayout, headRow, token, dim,
                                N, headDim)] = front * cos[tableFront]
                                - back * sin[tableFront];
                            output[ropeTensorOffset(
                                outputLayout, headRow, token, dim + half,
                                N, headDim)] = back * cos[tableBack]
                                + front * sin[tableBack];
                        }
                    }
                }
            };

            #pragma omp parallel for if(shouldParallel) schedule(static)
            for (int row = 0; row < totalRows; ++row) {
                const bool isQ = row < qRows;
                const int localRow = isQ ? row : row - qRows;
                applyHead(
                    isQ ? q : k,
                    isQ ? qOut : kOut,
                    isQ ? qNumHeads : kNumHeads,
                    localRow,
                    isQ ? qOutputLayout : kOutputLayout);
            }
        }

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        static void ropeApplyNativeFp16PairLayouts(
                const uint16_t *qStorage, const uint16_t *kStorage,
                uint16_t *qOutStorage, uint16_t *kOutStorage,
                const float *sin, const float *cos,
                int B, int qNumHeads, int kNumHeads, int N, int headDim,
                int sinB, int sinH, int hw, bool interleaved,
                RopeLayout inputLayout,
                RopeLayout qOutputLayout,
                RopeLayout kOutputLayout) {
            const __fp16 *q = reinterpret_cast<const __fp16 *>(qStorage);
            const __fp16 *k = reinterpret_cast<const __fp16 *>(kStorage);
            __fp16 *qOut = reinterpret_cast<__fp16 *>(qOutStorage);
            __fp16 *kOut = reinterpret_cast<__fp16 *>(kOutStorage);
            const int half = headDim / 2;
            const int prefix = N - hw;
            const int qRows = B * qNumHeads;
            const int totalRows = qRows + B * kNumHeads;
            const long long totalElements =
                (long long)totalRows * N * headDim;

            #pragma omp parallel for if(totalElements >= kMinParallelElements) schedule(static)
            for (int row = 0; row < totalRows; ++row) {
                const bool isQ = row < qRows;
                const int headRow = isQ ? row : row - qRows;
                const int numHeads = isQ ? qNumHeads : kNumHeads;
                const __fp16 *input = isQ ? q : k;
                __fp16 *output = isQ ? qOut : kOut;
                const RopeLayout outputLayout =
                    isQ ? qOutputLayout : kOutputLayout;
                const int batch = headRow / numHeads;
                const int head = headRow % numHeads;
                const int tableHead =
                    (sinB == 1 ? 0 : batch) * sinH
                    + (sinH == 1 ? 0 : head);
                for (int token = 0; token < N; ++token) {
                    if (token < prefix) {
                        for (int dim = 0; dim < headDim; ++dim) {
                            output[ropeTensorOffset(
                                outputLayout, headRow, token, dim,
                                N, headDim)] = input[ropeTensorOffset(
                                inputLayout, headRow, token, dim,
                                N, headDim)];
                        }
                        continue;
                    }
                    const int ropeToken = token - prefix;
                    const int pairCount = interleaved ? headDim / 2 : half;
                    for (int pair = 0; pair < pairCount; ++pair) {
                        const int dim0 = interleaved ? pair * 2 : pair;
                        const int dim1 = interleaved ? dim0 + 1 : pair + half;
                        const __fp16 x0 = input[ropeTensorOffset(
                            inputLayout, headRow, token, dim0,
                            N, headDim)];
                        const __fp16 x1 = input[ropeTensorOffset(
                            inputLayout, headRow, token, dim1,
                            N, headDim)];
                        const long long table0 = ropeTableOffset(
                            inputLayout, tableHead, ropeToken, dim0,
                            hw, headDim);
                        const long long table1 = ropeTableOffset(
                            inputLayout, tableHead, ropeToken, dim1,
                            hw, headDim);
                        output[ropeTensorOffset(
                            outputLayout, headRow, token, dim0,
                            N, headDim)] = x0 * (__fp16)cos[table0]
                            - x1 * (__fp16)sin[table0];
                        output[ropeTensorOffset(
                            outputLayout, headRow, token, dim1,
                            N, headDim)] = x1 * (__fp16)cos[table1]
                            + x0 * (__fp16)sin[table1];
                    }
                }
            }
        }
#endif

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

        static void ropeApplyUint8PairLayouts(
                const uint8_t *q, const uint8_t *k,
                uint8_t *qOut, uint8_t *kOut,
                const float *sin, const float *cos,
                int B, int qNumHeads, int kNumHeads, int N, int headDim,
                int sinB, int sinH, int hw,
                const Uint8RopeQuant &qQuant,
                const Uint8RopeQuant &kQuant,
                bool interleaved,
                RopeLayout inputLayout,
                RopeLayout qOutputLayout,
                RopeLayout kOutputLayout) {
            const int half = headDim / 2;
            const int prefix = N - hw;
            const int qRows = B * qNumHeads;
            const int totalRows = qRows + B * kNumHeads;
            const long long totalElements =
                (long long)totalRows * N * headDim;

            #pragma omp parallel for if(totalElements >= kMinParallelElements) schedule(static)
            for (int row = 0; row < totalRows; ++row) {
                const bool isQ = row < qRows;
                const int headRow = isQ ? row : row - qRows;
                const int numHeads = isQ ? qNumHeads : kNumHeads;
                const uint8_t *input = isQ ? q : k;
                uint8_t *output = isQ ? qOut : kOut;
                const Uint8RopeQuant &quant = isQ ? qQuant : kQuant;
                const RopeLayout outputLayout =
                    isQ ? qOutputLayout : kOutputLayout;
                const int batch = headRow / numHeads;
                const int head = headRow % numHeads;
                const int tableHead =
                    (sinB == 1 ? 0 : batch) * sinH
                    + (sinH == 1 ? 0 : head);
                for (int token = 0; token < N; ++token) {
                    if (token < prefix) {
                        for (int dim = 0; dim < headDim; ++dim) {
                            const int centered =
                                (int)input[ropeTensorOffset(
                                    inputLayout, headRow, token, dim,
                                    N, headDim)]
                                - quant.inputZeroPoint;
                            output[ropeTensorOffset(
                                outputLayout, headRow, token, dim,
                                N, headDim)] = quantizeCenteredScalar(
                                    (float)centered, quant);
                        }
                        continue;
                    }
                    const int ropeToken = token - prefix;
                    const int pairCount = interleaved ? headDim / 2 : half;
                    for (int pair = 0; pair < pairCount; ++pair) {
                        const int dim0 = interleaved ? pair * 2 : pair;
                        const int dim1 = interleaved ? dim0 + 1 : pair + half;
                        const float x0 = (float)(
                            (int)input[ropeTensorOffset(
                                inputLayout, headRow, token, dim0,
                                N, headDim)] - quant.inputZeroPoint);
                        const float x1 = (float)(
                            (int)input[ropeTensorOffset(
                                inputLayout, headRow, token, dim1,
                                N, headDim)] - quant.inputZeroPoint);
                        const long long table0 = ropeTableOffset(
                            inputLayout, tableHead, ropeToken, dim0,
                            hw, headDim);
                        const long long table1 = ropeTableOffset(
                            inputLayout, tableHead, ropeToken, dim1,
                            hw, headDim);
                        output[ropeTensorOffset(
                            outputLayout, headRow, token, dim0,
                            N, headDim)] = quantizeCenteredScalar(
                                x0 * cos[table0] - x1 * sin[table0], quant);
                        output[ropeTensorOffset(
                            outputLayout, headRow, token, dim1,
                            N, headDim)] = quantizeCenteredScalar(
                                x1 * cos[table1] + x0 * sin[table1], quant);
                    }
                }
            }
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
            const bool inputSequenceLast =
                param->inputLayout == RopeLayout::BHDN;

            int B = qShape[0];
            int qNumHeads = qShape[1];
            int kNumHeads = kShape[1];
            int N = qShape[inputSequenceLast ? 3 : 2];
            int headDim = qShape[inputSequenceLast ? 2 : 3];
            int hw = sinShape[
                inputSequenceLast
                    ? sinShape.size() - 1
                    : sinShape.size() - 2];
            int sinB = sinShape[0];
            int sinH = sinShape[1];
            const bool defaultFastLayout =
                param->inputLayout == RopeLayout::BHND
                && param->qOutputLayout == RopeLayout::BHND
                && param->kOutputLayout == RopeLayout::BHND;

            // sin and cos are always float
            const float *sinPtr = (const float *) GetTensordata(sinData);
            const float *cosPtr = (const float *) GetTensordata(cosData);

            if (GetTensorType(qData) == TFCAPI_FLOAT && GetTensorType(kData) == TFCAPI_FLOAT) {
                // Pure float path
                const float *qPtr = (const float *) GetTensordata(qData);
                const float *kPtr = (const float *) GetTensordata(kData);
                float *qOutPtr = (float *) GetTensordata(qOutData);
                float *kOutPtr = (float *) GetTensordata(kOutData);

                if (defaultFastLayout) {
                    ropeApplyFloatPair(
                        qPtr, kPtr, qOutPtr, kOutPtr, sinPtr, cosPtr,
                        B, qNumHeads, kNumHeads, N, headDim,
                        sinB, sinH, hw,
                        param->useFp16, param->interleaved);
                } else {
                    ropeApplyFloatPairLayouts(
                        qPtr, kPtr, qOutPtr, kOutPtr, sinPtr, cosPtr,
                        B, qNumHeads, kNumHeads, N, headDim,
                        sinB, sinH, hw, param->interleaved,
                        param->inputLayout,
                        param->qOutputLayout,
                        param->kOutputLayout);
                }

            } else if (GetTensorType(qData) == TFCAPI_FLOAT16
                    && GetTensorType(kData) == TFCAPI_FLOAT16) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                if (defaultFastLayout) {
                    ropeApplyNativeFp16Pair(
                        (const uint16_t *)GetTensordata(qData),
                        (const uint16_t *)GetTensordata(kData),
                        (uint16_t *)GetTensordata(qOutData),
                        (uint16_t *)GetTensordata(kOutData),
                        sinPtr, cosPtr,
                        B, qNumHeads, kNumHeads, N, headDim,
                        sinB, sinH, hw, param->interleaved);
                } else {
                    ropeApplyNativeFp16PairLayouts(
                        (const uint16_t *)GetTensordata(qData),
                        (const uint16_t *)GetTensordata(kData),
                        (uint16_t *)GetTensordata(qOutData),
                        (uint16_t *)GetTensordata(kOutData),
                        sinPtr, cosPtr,
                        B, qNumHeads, kNumHeads, N, headDim,
                        sinB, sinH, hw, param->interleaved,
                        param->inputLayout,
                        param->qOutputLayout,
                        param->kOutputLayout);
                }
#else
                throw std::runtime_error(
                    "ApplyRope native FP16 tensors require ARMv8.2 FP16");
#endif
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

                if (defaultFastLayout) {
                    ropeApplyUint8Pair(
                        qUint8, kUint8, qOutUint8, kOutUint8,
                        sinPtr, cosPtr,
                        B, qNumHeads, kNumHeads, N, headDim,
                        sinB, sinH, hw,
                        qRopeQuant, kRopeQuant, param->interleaved);
                } else {
                    ropeApplyUint8PairLayouts(
                        qUint8, kUint8, qOutUint8, kOutUint8,
                        sinPtr, cosPtr,
                        B, qNumHeads, kNumHeads, N, headDim,
                        sinB, sinH, hw,
                        qRopeQuant, kRopeQuant, param->interleaved,
                        param->inputLayout,
                        param->qOutputLayout,
                        param->kOutputLayout);
                }
            } else {
                throw std::runtime_error(
                    "ApplyRope requires matching FP32, FP16, or UINT8 Q/K tensors");
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

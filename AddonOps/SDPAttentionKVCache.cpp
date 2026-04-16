//
// SDPAttentionKVCache 自定义算子 — 带外部 KV Cache 的缩放点积注意力
// 用于 Falcon-Perception 的自回归推理
//
// 实现功能:
//   1. 将新的 K, V 插入到 KV Cache 的指定位置
//   2. 计算缩放点积注意力 (Q * K^T / sqrt(d))
//   3. 应用因果掩码 (causal mask)
//   4. Softmax 归一化
//   5. 加权求和 (attn_weights * V)
//   6. 可选: Attention Sinks (sigmoid(lse - sinks) 门控)
//
// 输入:
//   [0] Q:          [B, n_heads_q, S_q, head_dim]    查询
//   [1] K_new:      [B, n_heads_kv, S_q, head_dim]   新的键 (用于插入缓存)
//   [2] V_new:      [B, n_heads_kv, S_q, head_dim]   新的值 (用于插入缓存)
//   [3] K_cache:    [B, n_heads_kv, max_seq, head_dim] 键缓存
//   [4] V_cache:    [B, n_heads_kv, max_seq, head_dim] 值缓存
//   [5] cache_pos:  [1] int32  当前缓存位置 (起始写入位置)
//   [6] sinks:      [n_heads_q] float  注意力汇参数 (可选, 全零表示不使用)
//
// 输出:
//   [0] attn_output: [B, n_heads_q, S_q, head_dim]  注意力输出
//   [1] K_cache_updated: [B, n_heads_kv, max_seq, head_dim] 更新后的键缓存
//   [2] V_cache_updated: [B, n_heads_kv, max_seq, head_dim] 更新后的值缓存
//
// JSON 参数:
//   "nRep": int,  GQA 中 KV head 重复次数 (n_heads_q / n_heads_kv)
//   "useSinks": bool, 是否使用 attention sinks
//

#ifndef NPU40T_SDP_ATTENTION_KV_CACHE_H
#define NPU40T_SDP_ATTENTION_KV_CACHE_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>

using namespace TFDL_CAPI;

namespace TFDLOP {
    namespace SDPAttentionKVCache {
        struct SDPAttnParam {
            int nRep = 1;        // GQA 重复次数
            bool useSinks = false;
        };

        void Prepare(TFContext tfContext, TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node), err);
            if (!err.empty()) {
                printf("[SDPAttentionKVCache] JSON parse error: %s\n", err.c_str());
            }

            SDPAttnParam *p = new SDPAttnParam();
            p->nRep = param["nRep"].int_value();
            if (p->nRep <= 0) p->nRep = 1;
            p->useSinks = param["useSinks"].bool_value();

            FreeNodeCustomParam(node, [](void *cp) { delete (SDPAttnParam *)cp; });
            NewNodeCustomParam(node, [&p]() -> void * { return p; });
        }

        void Reshape(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            TFCHECK_GE(info.InputNames.size(), 6);
            TFCHECK_EQ(info.OutputNames.size(), 3);

            auto qData   = GetTensorByName(tfContext, info.InputNames[0]);
            auto kCache  = GetTensorByName(tfContext, info.InputNames[3]);
            auto vCache  = GetTensorByName(tfContext, info.InputNames[4]);
            auto outData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto kOut    = GetTensorByName(tfContext, info.OutputNames[1]);
            auto vOut    = GetTensorByName(tfContext, info.OutputNames[2]);

            // attn_output 形状与 Q 相同
            ReSizeTensor(outData, GetTensorShape(qData));
            SetTensorType(outData, GetTensorType(qData));

            // KV cache 输出形状与输入缓存相同
            ReSizeTensor(kOut, GetTensorShape(kCache));
            SetTensorType(kOut, GetTensorType(kCache));
            ReSizeTensor(vOut, GetTensorShape(vCache));
            SetTensorType(vOut, GetTensorType(vCache));
        }

        // sigmoid 辅助函数
        static float _sigmoid_float(float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        static void sdpAttentionFloat(
            const float *Q, const float *K_new, const float *V_new,
            const float *K_cache, const float *V_cache,
            float *attn_output, float *K_cache_out, float *V_cache_out,
            const float *sinks,
            int B, int nHeadsQ, int nHeadsKV, int Sq, int maxSeq, int headDim,
            int cachePos, int nRep, bool useSinks
        ) {
            // 1. 复制 KV cache 并插入新的 K/V
            int kvCacheSize = B * nHeadsKV * maxSeq * headDim;
            memcpy(K_cache_out, K_cache, kvCacheSize * sizeof(float));
            memcpy(V_cache_out, V_cache, kvCacheSize * sizeof(float));

            // 插入新的 K/V 到 cache_pos 位置
            for (int b = 0; b < B; b++) {
                for (int h = 0; h < nHeadsKV; h++) {
                    for (int s = 0; s < Sq; s++) {
                        int dstOffset = ((b * nHeadsKV + h) * maxSeq + cachePos + s) * headDim;
                        int srcOffset = ((b * nHeadsKV + h) * Sq + s) * headDim;
                        memcpy(K_cache_out + dstOffset, K_new + srcOffset, headDim * sizeof(float));
                        memcpy(V_cache_out + dstOffset, V_new + srcOffset, headDim * sizeof(float));
                    }
                }
            }

            // 2. 计算注意力
            int kvLen = cachePos + Sq;  // 有效 KV 长度
            float scale = 1.0f / sqrtf((float)headDim);

            for (int b = 0; b < B; b++) {
                for (int hq = 0; hq < nHeadsQ; hq++) {
                    // GQA: 映射 query head 到 kv head
                    int hk = hq / nRep;

                    for (int sq = 0; sq < Sq; sq++) {
                        const float *qVec = Q + ((b * nHeadsQ + hq) * Sq + sq) * headDim;

                        // 计算 Q * K^T / sqrt(d)
                        // 分配 attention scores
                        float *scores = new float[kvLen];
                        float maxScore = -1e30f;

                        for (int kv = 0; kv < kvLen; kv++) {
                            const float *kVec = K_cache_out +
                                ((b * nHeadsKV + hk) * maxSeq + kv) * headDim;

                            float dot = 0.0f;
                            for (int d = 0; d < headDim; d++) {
                                dot += qVec[d] * kVec[d];
                            }
                            scores[kv] = dot * scale;

                            // 因果掩码: 对于 prefill, 只允许 attend 到 sq 位置之前
                            // 对于 decode (sq==0), 可以 attend 到所有 kvLen
                            // 简化: 如果 sq > 0 (prefill), 应用因果掩码
                            if (Sq > 1 && kv > cachePos + sq) {
                                scores[kv] = -1e30f;
                            }
                            if (scores[kv] > maxScore) maxScore = scores[kv];
                        }

                        // Softmax
                        float sumExp = 0.0f;
                        float lse = maxScore;  // log-sum-exp
                        for (int kv = 0; kv < kvLen; kv++) {
                            scores[kv] = expf(scores[kv] - maxScore);
                            sumExp += scores[kv];
                        }
                        if (sumExp > 0) {
                            for (int kv = 0; kv < kvLen; kv++) {
                                scores[kv] /= sumExp;
                            }
                        }

                        // 计算真实的 log-sum-exp
                        lse = logf(sumExp > 0 ? sumExp : 1e-30f) + maxScore;

                        // 加权求和: attn_weights * V
                        float *outVec = attn_output + ((b * nHeadsQ + hq) * Sq + sq) * headDim;
                        memset(outVec, 0, headDim * sizeof(float));
                        for (int kv = 0; kv < kvLen; kv++) {
                            const float *vVec = V_cache_out +
                                ((b * nHeadsKV + hk) * maxSeq + kv) * headDim;
                            float w = scores[kv];
                            for (int d = 0; d < headDim; d++) {
                                outVec[d] += w * vVec[d];
                            }
                        }

                        // Attention Sinks: output *= sigmoid(lse - sinks[hq])
                        if (useSinks && sinks != nullptr) {
                            float sinkVal = _sigmoid_float(lse - sinks[hq]);
                            for (int d = 0; d < headDim; d++) {
                                outVec[d] *= sinkVal;
                            }
                        }

                        delete[] scores;
                    }
                }
            }
        }

        void Eval(TFContext tfContext, TFNode node) {
            SDPAttnParam *p = (SDPAttnParam *)GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);

            auto qData     = GetTensorByName(tfContext, info.InputNames[0]);
            auto kNewData  = GetTensorByName(tfContext, info.InputNames[1]);
            auto vNewData  = GetTensorByName(tfContext, info.InputNames[2]);
            auto kCacheData= GetTensorByName(tfContext, info.InputNames[3]);
            auto vCacheData= GetTensorByName(tfContext, info.InputNames[4]);
            auto posData   = GetTensorByName(tfContext, info.InputNames[5]);

            auto outData   = GetTensorByName(tfContext, info.OutputNames[0]);
            auto kOutData  = GetTensorByName(tfContext, info.OutputNames[1]);
            auto vOutData  = GetTensorByName(tfContext, info.OutputNames[2]);

            // 读取 cache_pos (标量 int)
            int cachePos = 0;
            void *posRaw = GetTensordata(posData);
            if (GetTensorType(posData) == TFCAPI_INT32) {
                cachePos = *((int32_t *)posRaw);
            } else if (GetTensorType(posData) == TFCAPI_INT64) {
                cachePos = (int)(*((int64_t *)posRaw));
            } else if (GetTensorType(posData) == TFCAPI_FLOAT) {
                cachePos = (int)(*((float *)posRaw));
            }

            auto qShape = GetTensorShape(qData);
            int B = qShape[0], nHeadsQ = qShape[1], Sq = qShape[2], headDim = qShape[3];
            auto kShape = GetTensorShape(kNewData);
            int nHeadsKV = kShape[1];
            auto cacheShape = GetTensorShape(kCacheData);
            int maxSeq = cacheShape[2];

            // 读取 sinks (可选第7个输入)
            const float *sinksPtr = nullptr;
            if (info.InputNames.size() >= 7) {
                auto sinksData = GetTensorByName(tfContext, info.InputNames[6]);
                sinksPtr = (const float *)GetTensordata(sinksData);
            }

            if (GetTensorType(qData) == TFCAPI_FLOAT) {
                sdpAttentionFloat(
                    (const float *)GetTensordata(qData),
                    (const float *)GetTensordata(kNewData),
                    (const float *)GetTensordata(vNewData),
                    (const float *)GetTensordata(kCacheData),
                    (const float *)GetTensordata(vCacheData),
                    (float *)GetTensordata(outData),
                    (float *)GetTensordata(kOutData),
                    (float *)GetTensordata(vOutData),
                    sinksPtr,
                    B, nHeadsQ, nHeadsKV, Sq, maxSeq, headDim,
                    cachePos, p->nRep, p->useSinks
                );
            } else if (GetTensorType(qData) == TFCAPI_UINT8) {
                // uint8 路径: 反量化所有输入 → float 计算 → 重新量化输出
                int qCount = B * nHeadsQ * Sq * headDim;
                int kvNewCount = B * nHeadsKV * Sq * headDim;
                int kvCacheCount = B * nHeadsKV * maxSeq * headDim;

                auto qQ = GetTensorQuantizeInfo(tfContext, info.InputNames[0]);
                auto kQ = GetTensorQuantizeInfo(tfContext, info.InputNames[1]);
                auto vQ = GetTensorQuantizeInfo(tfContext, info.InputNames[2]);
                auto kcQ = GetTensorQuantizeInfo(tfContext, info.InputNames[3]);
                auto vcQ = GetTensorQuantizeInfo(tfContext, info.InputNames[4]);

                float *qF = new float[qCount];
                float *kNewF = new float[kvNewCount];
                float *vNewF = new float[kvNewCount];
                float *kCacheF = new float[kvCacheCount];
                float *vCacheF = new float[kvCacheCount];

                DeQuantizeTensorData(qF, (uint8_t *)GetTensordata(qData), qCount, qQ);
                DeQuantizeTensorData(kNewF, (uint8_t *)GetTensordata(kNewData), kvNewCount, kQ);
                DeQuantizeTensorData(vNewF, (uint8_t *)GetTensordata(vNewData), kvNewCount, vQ);
                DeQuantizeTensorData(kCacheF, (uint8_t *)GetTensordata(kCacheData), kvCacheCount, kcQ);
                DeQuantizeTensorData(vCacheF, (uint8_t *)GetTensordata(vCacheData), kvCacheCount, vcQ);

                float *outF = new float[qCount];
                float *kCacheOutF = new float[kvCacheCount];
                float *vCacheOutF = new float[kvCacheCount];

                sdpAttentionFloat(
                    qF, kNewF, vNewF, kCacheF, vCacheF,
                    outF, kCacheOutF, vCacheOutF, sinksPtr,
                    B, nHeadsQ, nHeadsKV, Sq, maxSeq, headDim,
                    cachePos, p->nRep, p->useSinks
                );

                auto outQnt = GetTensorQuantizeInfo(tfContext, info.OutputNames[0]);
                auto kOutQnt = GetTensorQuantizeInfo(tfContext, info.OutputNames[1]);
                auto vOutQnt = GetTensorQuantizeInfo(tfContext, info.OutputNames[2]);

                QuantizeTensorData((uint8_t *)GetTensordata(outData), outF, qCount, outQnt);
                QuantizeTensorData((uint8_t *)GetTensordata(kOutData), kCacheOutF, kvCacheCount, kOutQnt);
                QuantizeTensorData((uint8_t *)GetTensordata(vOutData), vCacheOutF, kvCacheCount, vOutQnt);

                delete[] qF; delete[] kNewF; delete[] vNewF;
                delete[] kCacheF; delete[] vCacheF;
                delete[] outF; delete[] kCacheOutF; delete[] vCacheOutF;
            }
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *cp) { delete (SDPAttnParam *)cp; });
        }
    }
    RegistOp(SDPAttentionKVCache)
    .Set(SDPAttentionKVCache::Prepare, SDPAttentionKVCache::Reshape,
         SDPAttentionKVCache::Eval, SDPAttentionKVCache::Free);
}

#endif // NPU40T_SDP_ATTENTION_KV_CACHE_H

//
// Created for RopePositionEmbedding custom operator
// Generates sin/cos rotary position embeddings from learnable periods
//
// Inputs: H (scalar int32), W (scalar int64), periods [P] (float)
// Outputs: sin [HW, 4*P], cos [HW, 4*P]
//

#ifndef NPU40T_ROPE_POSITION_EMBEDDING_H
#define NPU40T_ROPE_POSITION_EMBEDDING_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <cstring>
#include <cassert>
#include <string>
using namespace TFDL_CAPI;
namespace TFDLOP {
    namespace RopePositionEmbedding {
        struct RopePosEmbParam {
            std::string normalizeCoords; // "min", "max", "separate"
        };

        void Prepare(TFContext tfContext, TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node), err);
            if (!err.empty()) {
                printf("%s\n", err.c_str());
            }

            RopePosEmbParam *p = new RopePosEmbParam();
            p->normalizeCoords = param["normalizeCoords"].string_value();
            if (p->normalizeCoords.empty()) {
                p->normalizeCoords = "separate";
            }

            FreeNodeCustomParam(node, [](void *customparam) {
                delete (RopePosEmbParam *) customparam;
            });
            NewNodeCustomParam(node, [&p]() -> void * {
                return p;
            });
        }

        // Helper: read an integer value from a scalar tensor of any int/float type
        static int readScalarInt(TFTensor tensor) {
            void *data = GetTensordata(tensor);
            if (!data) return 0;
            switch (GetTensorType(tensor)) {
                case TFCAPI_INT32:  return *((int32_t *) data);
                case TFCAPI_INT64:  return (int) *((int64_t *) data);
                case TFCAPI_FLOAT:  return (int) *((float *) data);
                default:            return 0;
            }
        }

        void Reshape(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            TFCHECK_EQ(info.InputNames.size(), 3);
            TFCHECK_EQ(info.OutputNames.size(), 2);

            auto HData = GetTensorByName(tfContext, info.InputNames[0]);                                                                                                                                                                     
            auto WData = GetTensorByName(tfContext, info.InputNames[1]);                                                                                                                                                                     
            auto periodsData = GetTensorByName(tfContext, info.InputNames[2]);
            auto sinOutData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto cosOutData = GetTensorByName(tfContext, info.OutputNames[1]);
            // H and W are scalar tensors — read their values for shape inference                                                                                                                                                            
            int H, W;                                                                                                                                                                                                                        
            if (GetTensorType(HData) == TFCAPI_INT32) {                                                                                                                                                                                      
                H = *((int32_t *) GetTensordata(HData));                                                                                                                                                                                     
            } else if (GetTensorType(HData) == TFCAPI_INT64) {                                                                                                                                                                               
                H = (int) *((int64_t *) GetTensordata(HData));                                                                                                                                                                               
            } else {                                                                                                                                                                                                                         
                H = (int) *((float *) GetTensordata(HData));                                                                                                                                                                                 
            }                                                                                                                                                                                                                                
            if (GetTensorType(WData) == TFCAPI_INT64) {                                                                                                                                                                                      
                W = (int) *((int64_t *) GetTensordata(WData));                                                                                                                                                                               
            } else if (GetTensorType(WData) == TFCAPI_INT32) {                                                                                                                                                                               
                W = *((int32_t *) GetTensordata(WData));                                                                                                                                                                                     
            } else {                                                                                                                                                                                                                         
                W = (int) *((float *) GetTensordata(WData));                                                                                                                                                                                 
            }                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                   
            auto periodsShape = GetTensorShape(periodsData);
            TFCHECK_EQ(periodsShape.size(), 1);
            int P = periodsShape[0];        // D_head // 4                                                                                                                                                                                   
            int D = 4 * P;                  // D_head                                                                                                                                                                                        
            int HW = H * W;                                                                                                                                                                                                                 
            vector<int> outShape = {HW, D};           
            ReSizeTensor(sinOutData, outShape);
            SetTensorType(sinOutData, TFCAPI_FLOAT);
            ReSizeTensor(cosOutData, outShape);
            SetTensorType(cosOutData, TFCAPI_FLOAT);
        }

        static void computeRopePosEmb(
                const float *periods, float *sinOut, float *cosOut,
                int H, int W, int P, int normalizeMode) {
            int D = 4 * P;
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float ch, cw;
                    if (normalizeMode == 0) {       // "min"
                        float minHW = (H < W) ? (float)H : (float)W;
                        ch = (0.5f + (float)h) / minHW;
                        cw = (0.5f + (float)w) / minHW;
                    } else if (normalizeMode == 1) { // "max"
                        float maxHW = (H > W) ? (float)H : (float)W;
                        ch = (0.5f + (float)h) / maxHW;
                        cw = (0.5f + (float)w) / maxHW;
                    } else {                         // "separate"
                        ch = (0.5f + (float)h) / (float)H;
                        cw = (0.5f + (float)w) / (float)W;
                    }
                    float coordH = 2.0f * ch - 1.0f;
                    float coordW = 2.0f * cw - 1.0f;

                    int idx = h * W + w;
                    for (int p = 0; p < P; p++) {
                        float angleH = 2.0f * (float)M_PI * coordH / periods[p];
                        float angleW = 2.0f * (float)M_PI * coordW / periods[p];

                        float sinH = sinf(angleH);
                        float cosH = cosf(angleH);
                        float sinW = sinf(angleW);
                        float cosW = cosf(angleW);

                        int base = idx * D;
                        sinOut[base + p]       = sinH;
                        sinOut[base + P + p]   = sinW;
                        sinOut[base + 2*P + p] = sinH;
                        sinOut[base + 3*P + p] = sinW;

                        cosOut[base + p]       = cosH;
                        cosOut[base + P + p]   = cosW;
                        cosOut[base + 2*P + p] = cosH;
                        cosOut[base + 3*P + p] = cosW;
                    }
                }
            }
        }

        void Eval(TFContext tfContext, TFNode node) {
            RopePosEmbParam *p = (RopePosEmbParam *) GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            auto HData = GetTensorByName(tfContext, info.InputNames[0]);
            auto WData = GetTensorByName(tfContext, info.InputNames[1]);
            auto periodsData = GetTensorByName(tfContext, info.InputNames[2]);
            auto sinOutData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto cosOutData = GetTensorByName(tfContext, info.OutputNames[1]);

            // Read H and W from scalar tensors
            int H, W;
            if (GetTensorType(HData) == TFCAPI_INT32) {
                H = *((int32_t *) GetTensordata(HData));
            } else if (GetTensorType(HData) == TFCAPI_INT64) {
                H = (int) *((int64_t *) GetTensordata(HData));
            } else {
                H = (int) *((float *) GetTensordata(HData));
            }
            if (GetTensorType(WData) == TFCAPI_INT64) {
                W = (int) *((int64_t *) GetTensordata(WData));
            } else if (GetTensorType(WData) == TFCAPI_INT32) {
                W = *((int32_t *) GetTensordata(WData));
            } else {
                W = (int) *((float *) GetTensordata(WData));
            }

            int P = GetTensorShape(periodsData)[0];
            const float *periods = (const float *) GetTensordata(periodsData);
            float *sinOut = (float *) GetTensordata(sinOutData);
            float *cosOut = (float *) GetTensordata(cosOutData);

            int normalizeMode = 2;
            if (p->normalizeCoords == "min") normalizeMode = 0;
            else if (p->normalizeCoords == "max") normalizeMode = 1;

            computeRopePosEmb(periods, sinOut, cosOut, H, W, P, normalizeMode);
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *customparam) {
                delete (RopePosEmbParam *) customparam;
            });
        }
    }
    RegistOp(RopePositionEmbedding)
    .Set(RopePositionEmbedding::Prepare, RopePositionEmbedding::Reshape,
         RopePositionEmbedding::Eval, RopePositionEmbedding::Free);
}

#endif //NPU40T_ROPE_POSITION_EMBEDDING_H

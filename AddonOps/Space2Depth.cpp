//
// Created by test on 2020/6/12.
//


#ifndef NPU40T_POWER_H
#define NPU40T_POWER_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <assert.h>
#include <cstring>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
using namespace TFDL_CAPI;
namespace TFDLOP {
    namespace Space2Depth {
        struct Space2DepthParam{
            int stride_w;
            int stride_h;
        };
        void Prepare(TFContext tfContext, TFNode node) {
            auto custromparam = GetNodeCustomParam(node);
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node),err);
            if(!err.empty()){
                printf("%s\n",err.c_str());
            }
            Space2DepthParam *Param = new Space2DepthParam();
            Param->stride_h = param["stride_h"].int_value();
            Param->stride_w = param["stride_w"].int_value();

            FreeNodeCustomParam(node, [](void *custromparam) {
                delete (Space2DepthParam *) custromparam;
            });
            NewNodeCustomParam(node,[&Param]()->void*{
                return Param;
            });

        }
        void Reshape(TFContext tfContext, TFNode node) {
            Space2DepthParam *param = (Space2DepthParam *) GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            auto inputData = GetTensorByName(tfContext, info.InputNames[0]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto shape = GetTensorShape(inputData);
            // w or h must be mod by stride
            assert(shape[2]%param->stride_h == 0);
            assert(shape[3]%param->stride_w == 0);
            vector<int> newshape = shape;
            newshape[1] *= param->stride_h*param->stride_w;
            newshape[2] /= param->stride_h;
            newshape[3] /= param->stride_w;
            ReSizeTensor(outputData,newshape);
            SetTensorType(outputData,GetTensorType(inputData));
        }

        template<typename T>
        void Expand(T* input,T* output,vector<int>ishape,vector<int>oshape,int stride_h,int stride_w){
            int icountc = ishape[1]*ishape[2]*ishape[3];
            int icounth = ishape[2]*ishape[3];
            int icountw = ishape[3];
            int ocountc = oshape[1]*oshape[2]*oshape[3];
            int ocounth = oshape[2]*oshape[3];
            int ocountw = oshape[3];
            int ichannel = ishape[1];
            //int channelAll = stride_h*stride_w;
            for(int b = 0;b<ishape[0];b++){
                for(int c=0;c<ishape[1];c++){
                    for(int hs=0;hs<stride_h;hs++){
                        for(int h=hs;h<ishape[2];h+=stride_h) {
                            T * tempinput = input + b*icountc + c*icounth + h*icountw;
                            switch (stride_w){
                                case(1):{
                                    memcpy(output,tempinput, sizeof(T)*oshape[3]);
                                    output += oshape[3];
                                    break;
                                }
#ifdef __aarch64__
                                case(2): {
                                    int w = 0;
                                    if (std::is_same<uint8_t, T>::value) {
                                        for (; w + 16 <= ishape[3]; w += 16) {
                                            uint8x8x2_t idata = vld2_u8((uint8_t *) tempinput + w);
                                            /*vst1_u8((uint8_t *) output + b * ocountc +
                                                    (hs * stride_w + 0 + c * channelAll) * ocounth +
                                                    h / stride_h * ocountw + w / stride_w, idata.val[0]);
                                            vst1_u8((uint8_t *) output + b * ocountc +
                                                    (hs * stride_w + 1 + c * channelAll) * ocounth +
                                                    h / stride_h * ocountw + w / stride_w, idata.val[1]);*/
                                            vst1_u8((uint8_t *) output + b * ocountc +
                                                    (ichannel*(hs  + 0) + c) * ocounth +
                                                    h / stride_h * ocountw + w / stride_w, idata.val[0]);
                                            vst1_u8((uint8_t *) output + b * ocountc +
                                                    (ichannel*(hs  + 1*stride_h) + c) * ocounth +
                                                    h / stride_h * ocountw + w / stride_w, idata.val[1]);
                                        }
                                    } else if (std::is_same<float, T>::value) {
                                        for (; w + 8 <= ishape[3]; w += 8) {
                                            float32x4x2_t idata = vld2q_f32((float *) tempinput + w);
                                            /*vst1q_f32((float *) output + b * ocountc +
                                                      (hs * stride_w + 0 + c * channelAll) * ocounth +
                                                      h / stride_h * ocountw + w / stride_w, idata.val[0]);
                                            vst1q_f32((float *) output + b * ocountc +
                                                      (hs * stride_w + 1 + c * channelAll) * ocounth +
                                                      h / stride_h * ocountw + w / stride_w, idata.val[1]);*/
                                            vst1q_f32((float *) output + b * ocountc +
                                                      ((hs  + 0)*ichannel + c) * ocounth +
                                                      h / stride_h * ocountw + w / stride_w, idata.val[0]);
                                            vst1q_f32((float *) output + b * ocountc +
                                                      ((hs  + 1*stride_h)*ichannel + c) * ocounth +
                                                      h / stride_h * ocountw + w / stride_w, idata.val[1]);

                                        }
                                    }
                                    for (; w < ishape[3]; w += stride_w) {
                                        int ws = w%2;
                                        //output[b * ocountc +((hs * stride_w + ws)  + c*channelAll) * ocounth + h / stride_h * ocountw + w / stride_w] = tempinput[w];
                                        output[b * ocountc +((hs+ws*stride_h)*ichannel+c) * ocounth + h / stride_h * ocountw + w / stride_w] = tempinput[w];
                                    }

                                    break;
                                }
#endif
                                default:{
                                    for (int ws = 0; ws < stride_w; ws++) {
                                        int w = ws;
                                        for(;w<ishape[3];w+=stride_w){
                                            //output[b*ocountc+(hs*stride_w+ws+c*channelAll)*ocounth+h/stride_h*ocountw+w/stride_w] = tempinput[w];
                                            output[b*ocountc+((hs+ws*stride_h)*ichannel+c)*ocounth+h/stride_h*ocountw+w/stride_w] = tempinput[w];
                                        }
                                    }
                                    break;
                                }
                            }

                        }
                    }
                }
            }
        }

        void Eval(TFContext tfContext, TFNode node) {
            Space2DepthParam *param = (Space2DepthParam *) GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            auto inputData = GetTensorByName(tfContext, info.InputNames[0]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto shape = GetTensorShape(inputData);
            auto oshape = GetTensorShape(outputData);
            if (GetTensorType(inputData) == TFCAPI_UINT8 && GetTensorType(outputData) == TFCAPI_UINT8) {
                uint8_t *input = (uint8_t *) GetTensordata(inputData);
                uint8_t *output = (uint8_t *) GetTensordata(outputData);
                Expand(input,output,shape,oshape,param->stride_h,param->stride_w);
            } else if (GetTensorType(inputData) == TFCAPI_FLOAT && GetTensorType(outputData) == TFCAPI_FLOAT) {
                float *input = (float *) GetTensordata(inputData);
                float *output = (float *) GetTensordata(outputData);
                Expand(input,output,shape,oshape,param->stride_h,param->stride_w);
            }
        }
        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *custromparam) {
                delete (Space2DepthParam *) custromparam;
            });
        }
    }
    RegistOp(Space2Depth)
    .Set(Space2Depth::Prepare,Space2Depth::Reshape,Space2Depth::Eval,Space2Depth::Free);
}

#endif //NPU40T_POWER_H

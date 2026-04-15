//
// Created by test on 2019/11/1.
//

#ifndef NPU40T_POWER_H
#define NPU40T_POWER_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cmath>
#include <cstring>
using namespace TFDL_CAPI;
namespace TFDLOP {
    namespace Power {
        struct PowerParam{
            float shift = -1;
            float scale = 0.00784313678741455;
            float power = 1;
            uint8_t maptable[256];
            bool isEqual = false;
        };
        void Prepare(TFContext tfContext, TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node),err);
            if(!err.empty()){
                printf("%s\n",err.c_str());
            }
            PowerParam *powerParam = new PowerParam();
            if(param["powerParameter"].is_null()){
                powerParam->shift = param["shift"].number_value();
                powerParam->scale = param["scale"].number_value();
                powerParam->power = param["power"].number_value();
            }
            else{
                powerParam->shift = param["powerParameter"]["shift"].number_value();
                powerParam->scale = param["powerParameter"]["scale"].number_value();
                powerParam->power = param["powerParameter"]["power"].number_value();
            }
            // if Prepare is used again, we need Free Param.
            FreeNodeCustomParam(node, [](void *custromparam) {
                delete (PowerParam *) custromparam;
            });
            NewNodeCustomParam(node,[&powerParam]()->void*{
                return powerParam;
            });

        }
        void Reshape(TFContext tfContext, TFNode node) {
            PowerParam *powerparam = (PowerParam *) GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            auto inputData = GetTensorByName(tfContext, info.InputNames[0]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);
            ReSizeTensor(outputData,GetTensorShape(inputData));
            SetTensorType(outputData,GetTensorType(inputData));
            if(GetTensorType(inputData) == TFCAPI_UINT8 && GetTensorType(outputData) == TFCAPI_UINT8){
                auto inputconfig = GetTensorQuantizeInfo(tfContext,info.InputNames[0]);
                auto outputconfig = GetTensorQuantizeInfo(tfContext,info.OutputNames[0]);
                powerparam->isEqual = true;
                for(int i=0;i<256;i++){
                    float in;
                    uint8_t j = i;
                    DeQuantizeTensorData(&in,&j,1, inputconfig);
                    float out = std::pow(powerparam->shift + powerparam->scale * in, powerparam->power);
                    QuantizeTensorData (&powerparam->maptable[i],&out,1, outputconfig);
                    powerparam->isEqual &= (powerparam->maptable[i] == i);
                }


            }
        }
        void Eval(TFContext tfContext, TFNode node) {
            PowerParam *powerparam = (PowerParam *) GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            auto inputData = GetTensorByName(tfContext, info.InputNames[0]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);
            int count = GetTensorCount(inputData,0);
            if (GetTensorType(inputData) == TFCAPI_UINT8 && GetTensorType(outputData) == TFCAPI_UINT8) {
                uint8_t * input = (uint8_t*)GetTensordata(inputData);
                uint8_t * output = (uint8_t*)GetTensordata(outputData);
                if (powerparam->isEqual) {
                    if (input != output) {
                        memcpy(output, input, count);
                    }

                    return;
                }
                SimpleMapdata(output,input,powerparam->maptable,count);
            } else if(GetTensorType(inputData) == TFCAPI_FLOAT && GetTensorType(outputData) == TFCAPI_FLOAT){
                float * input = (float*)GetTensordata(inputData);
                float * output = (float*)GetTensordata(outputData);
                for (int i = 0; i < count; i++) {
                    *(output++) = std::pow(powerparam->shift + powerparam->scale * *(input++), powerparam->power);
                }
            }
        }
        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node,[](void*custromparam){
                delete (PowerParam *) custromparam;
            });
        }
    }
    RegistOp(Power)
    .Set(Power::Prepare,Power::Reshape,Power::Eval,Power::Free);
}

#endif //NPU40T_POWER_H

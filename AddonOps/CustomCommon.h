//
// Created by test on 2019/11/14.
//

#ifndef NPU40T_CUSTOMCOMMON_H
#define NPU40T_CUSTOMCOMMON_H
#include "TFDL2_Common.h"
#include "TFDL2_C_API.h"
#define TFCHECK_EQ(a, b) assert(a == b);
#define TFCHECK_GT(a, b) assert(a > b);
#define TFCHECK_LT(a, b) assert(a < b);
#define TFCHECK_NE(a, b) assert(a != b);
#define TFCHECK_LE(a, b) assert(a <= b);
#define TFCHECK_GE(a, b) assert(a >= b);
#define TFDL2_CUSTOM_CAPI_EXPORT extern "C" __attribute__((visibility("default")))
#define TFDL_ATTRIBUTE_UNUSED __attribute__((unused))
#define RegistOpDefine(Opname)\
    TFDL2_CUSTOM_CAPI_EXPORT TFCAPIOP RegisterCustom##Opname();
static TFDL_CAPI::TFCAPIOP& TFCAPI_REGISTERCUSTOMOP(string CustomOpName){
    return TFDL_CAPI::RegisterCustomOp(CustomOpName);
}
#define RegistOp(Opname)\
    static TFDL_ATTRIBUTE_UNUSED TFCAPIOP& TFDLRegisterCustom##Opname = \
        TFCAPI_REGISTERCUSTOMOP(#Opname)\

#endif //NPU40T_CUSTOMCOMMON_H

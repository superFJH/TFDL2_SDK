//
// Created by test on 2019/11/12.
//

#ifndef NPU40T_TFDL2_COMMON_H
#define NPU40T_TFDL2_COMMON_H
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <TFLog.h>
#include <atomic>
using std::string;
using std::vector;
using std::map;
namespace TFDL_CAPI {
//C_API_TYPE
    class _tfExecutor;

    class _tfContext;

    class _tfNode;

    class _tfTensor;

    class _quantization;

    class _tfCalibration;

    class _tfServer;

    class _tfVision;

#define TFDL2_CAPI_EXPORT __attribute__((visibility("default")))

    template<typename T>
    class TFDL2_CAPI_EXPORT Object {
    public:
        Object() {
            self = nullptr;
        };

        Object(void *ptr, bool ref=true);

        ~Object();

        Object(const Object &object) noexcept;

        Object(Object &&object) noexcept;

        void operator=(const Object &object);

        void operator=(Object &&object);

        bool operator==(Object<T> ob2);

        bool IsValid();

    private:
        void *self;
        //the reference count for every ptr, just like shared_ptr;
        std::atomic_long* referenceCount = NULL;
        // this function can't be used by User;
        template<typename TFDL_TYPE, typename C_API_TYPE>
        friend TFDL_TYPE *GetBase(Object<C_API_TYPE> &object);
    };


    typedef class Object<_tfExecutor> TFExecutor;
    typedef class Object<_tfContext> TFContext;
    typedef class Object<_tfNode> TFNode;
    typedef class Object<_tfTensor> TFTensor;
    typedef class Object<_quantization> Quantization;
    typedef class Object<_tfCalibration> TFCalibration;
    typedef class Object<_tfServer> TFServer;
    typedef class Object<_tfVision> TFVision;
    typedef enum TFCAPI_DATATYPE {
        TFCAPI_UNKNOW = 0,
        TFCAPI_UINT8 = 1,
        TFCAPI_INT32 = 2,
        TFCAPI_FLOAT = 3,
        TFCAPI_UINT16 = 4,
        TFCAPI_BFLOAT16 = 5,
        TFCAPI_INT64 = 6,
        TFCAPI_FLOAT64 = 7,
        TFCAPI_STRING = 8,
        TFCAPI_FLOAT16 = 9,
        TFCAPI_INT16 = 10
    } TFCAPI_DATATYPE;

    typedef union {
        int64_t i64;
        int32_t i32;
        float f;
        uint8_t u8;
        uint16_t u16;
        double d;
    } DataUnion;

    enum TFConvertType {
        CAFFE2TFDL = 1,
        TENSORFLOW2TFDL = 2,
        TFLITE2TFDL = 3,
        ONNX2TFDL = 4
    };
    enum Device {
        CPU = 0,
        NPU10t = 1,
        NPU40t = 1
    };
    enum TFCalibrationMode {
        TFCali_Naive = 0,
        TFCali_KLD = 1,
        TFCali_MEAN = 2,
        TFCali_COVERAGE = 3,
    };
    typedef struct TFCAPIOP {
        TFCAPIOP() = delete;

        typedef std::function<void(TFContext , TFNode )> OpFunc;

        TFCAPIOP &Set(OpFunc prepare, OpFunc reshape, OpFunc eval, OpFunc free);
    } TFCAPIOP;
    typedef struct NodeInfo {
        vector<string> InputNames;
        vector<string> OutputNames;
        string OpName;
        string OpType;
    } NodeInfo;

    TFDL2_CAPI_EXPORT extern void popError(string msg);
}
#endif //NPU40T_TFDL2_COMMON_H

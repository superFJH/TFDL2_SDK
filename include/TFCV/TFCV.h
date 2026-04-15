//
// Created by test on 2019/10/29.
//

#ifndef NPU40T_TFCV_H
#define NPU40T_TFCV_H
#include "TFDL2_Common.h"
#include "TFDL2_C_API.h"
namespace TFDL_CAPI {
    namespace TFCV {
        enum DECODER_FLAGS {
            TFCV_BGR = 0,
            TFCV_RGB = 1,
            TFCV_Gray = 2,
            TFCV_I420 = 3
        };
        enum MediaTYPE{
            tJPEG = 0,
            tPNG = 1,
            tBMP = 2,
            tH264 = 3,
            tHEVC = 4
        };
        typedef std::tuple<int,int,int,int> CropSize;// {left,top,width,height}
        /*
         * flags: 0 : BGR
         *        1 : RGB
         *        2 : Gray
         */
        TFDL2_CAPI_EXPORT extern TFVision NewImgReader();

        TFDL2_CAPI_EXPORT extern TFVision NewVideoReader(int type=0,std::string decoderpath="");

        TFDL2_CAPI_EXPORT extern bool OpenURL(TFVision& tfVision,string path);

        TFDL2_CAPI_EXPORT extern bool OpenSource(TFVision& tfVision,void* data,size_t len);

        TFDL2_CAPI_EXPORT extern bool Close(TFVision& tfVision);

        TFDL2_CAPI_EXPORT extern bool VisionCopy(TFVision dst, TFVision src);

        TFDL2_CAPI_EXPORT extern int Compress(TFVision& src, const string& ext,vector<uint8_t>& dst,float scale=1.0);

        TFDL2_CAPI_EXPORT extern int DumpImgData(TFVision& tfVision,vector<uint8_t>& dst,DECODER_FLAGS flags = TFCV_BGR,float scale=1.0,CropSize cropSize=CropSize(-1,-1,-1,-1));

        TFDL2_CAPI_EXPORT extern int DumpImgData(TFVision& tfVision,uint8_t* dst,DECODER_FLAGS flags = TFCV_BGR,float scale=1.0,CropSize cropSize=CropSize(-1,-1,-1,-1));

        TFDL2_CAPI_EXPORT extern int DumpImgData(TFVision& tfVision,TFVision& dst,float scale=1.0,CropSize cropSize=CropSize(-1,-1,-1,-1));

        TFDL2_CAPI_EXPORT extern int DumpImgData(TFVision& tfVision,TFTensor& tfTensor,int batch=0,DECODER_FLAGS flags = TFCV_BGR,CropSize cropSize=CropSize(-1,-1,-1,-1));

        /*
         * read img to TFTensor
         * @param tfVision: the owner of decoder
         * @param dst: the target TFTensor
         * @param flags : the data format
         * @return:  1: Success; 0: Stop(only for videostream); -1: Error;
         */
        TFDL2_CAPI_EXPORT extern int ReadFrame(TFVision& tfVision);

        TFDL2_CAPI_EXPORT extern int GetWidth(TFVision& tfVision);

        TFDL2_CAPI_EXPORT extern int GetHeight(TFVision& tfVision);

        TFDL2_CAPI_EXPORT extern int GetFps(TFVision& tfVision);

        TFDL2_CAPI_EXPORT extern void SetFps(TFVision& tfVision,int customFps);

        TFDL2_CAPI_EXPORT extern void TFCV_INIT();

        TFDL2_CAPI_EXPORT extern MediaTYPE GetType(TFVision& tfVision);
    }
}
#endif //NPU40T_TFCV_H

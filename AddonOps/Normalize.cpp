//
// Created by test on 2019/10/14.
//

#ifndef TFDL2_NORMALIZE_H
#define TFDL2_NORMALIZE_H
/*namespace TFDL{
    namespace TFDLOP {
        namespace Normalize {
            struct NormalizeParam{
                bool across_spatial = false;
                bool channel_shared = false;
            };
            void Prepare(TFDL::TFContext *tfContext, TFDL::Node *node){
                auto param = node->GetParam();
                if(param->first!=TFDL)return;
                Attr::CustomParam* customParam = (Attr::CustomParam*)param->second;
                switch (param->first) {
                    case (CAFFE): {
                        caffe::LayerParameter layerParameter;
                        layerParameter.ParseFromString(customParam->protoStr);
                        NormalizeParam *normalizeParam = new NormalizeParam();
                        normalizeParam->across_spatial = layerParameter.norm_param().across_spatial();
                        normalizeParam->channel_shared = layerParameter.norm_param().channel_shared();
                        if(node->CustomParam != nullptr)delete (NormalizeParam*)node->CustomParam;
                        node->CustomParam = normalizeParam;
                        break;
                    }
                }
            }
            void Reshape(TFDL::TFContext *tfContext, TFDL::Node *node){
                auto input = tfContext->DataManager.GetDataPtr(node->GetInputNames()[0]);
                auto outputData = tfContext->DataManager.GetDataPtr(node->GetOutputNames()[0]);
                outputData->Resize(input->GetDims());
                outputData->SetType(input->GetType());
            }
            void Eval(TFDL::TFContext *tfContext, TFDL::Node *node){
                auto inputData = tfContext->DataManager.GetDataPtr(node->GetInputNames()[0]);
                auto outputData = tfContext->DataManager.GetDataPtr(node->GetOutputNames()[0]);
                int num = inputData->GetDim(0);
                int channels = inputData->GetDim(1);
                int spatial = inputData->Count(2);
                float* scale;
                if(node->GetWeightNames().empty()){
                    scale = new float[channels];
                    std::fill(scale,scale+channels,1.f);
                }else {
                    scale = tfContext->WeightManager->GetDataPtr(node->GetWeightNames()[0])->GetCpuData().f;
                }

                if (inputData->GetType() == TFDtypeUint8) {
                    Quantization* inputConfig = tfContext->DataManager.GetQuantizationConfig(inputData->GetName());
                    Quantization* outputConfig = tfContext->DataManager.GetQuantizationConfig(outputData->GetName());
                    uint8_t * input = inputData->GetCpuData().u8;
                    uint8_t * output = outputData->GetCpuData().u8;

                    int inputZero = inputConfig->zeroPoint[0];
                    for (int i = 0; i < num; i++) {
                        for (int j = 0; j < spatial; j++) {
                            int s = 0;
                            for (int c = 0; c < channels; c++) {
                                int y = input[c * spatial + j] - inputZero;
                                s += y * y;
                            }

                            float sum = sqrt((float)s * inputConfig->scale[0] * inputConfig->scale[0]);
                            for (int c = 0; c < channels; c++) {
                                output[c * spatial + j] = Math::Quantize(
                                        Math::Dequantize(input[c * spatial + j],inputConfig->scale[0],inputConfig->zeroPoint[0]) *
                                        (scale[c] / sum),outputConfig->scale[0],outputConfig->zeroPoint[0]);
                            }
                        }

                        input += (spatial * channels);
                        output += (spatial * channels);
                    }
                } else {
                    float * input = inputData->GetCpuData().f;
                    float * output = outputData->GetCpuData().f;
                    for (int i = 0; i < num; i++) {
                        for (int j = 0; j < spatial; j++) {
                            float sum = 0.0;
                            for (int c = 0; c < channels; c++) {
                                float x = input[c * spatial + j];
                                sum += x * x;
                            }

                            sum = sqrt(sum);
                            for (int c = 0; c < channels; c++) {
                                output[c * spatial + j] = input[c * spatial + j] * (scale[c] / sum);
                            }
                        }

                        input += (spatial * channels);
                        output += (spatial * channels);
                    }
                }
                if(node->GetWeightNames().empty()){
                    delete[] scale;
                }
            }
            void Free(TFDL::TFContext *tfContext, TFDL::Node *node){
                if(node->CustomParam != nullptr)delete (NormalizeParam*)node->CustomParam;
                node->CustomParam = nullptr;
            }
        }
    }
}*/
#endif //TFDL2_NORMALIZE_H

//
// Created by test on 2019/10/14.
//

#ifndef TFDL2_PRIORBOX_H
#define TFDL2_PRIORBOX_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include "bbox_util.h"
#include "cmath"
using namespace TFDL_CAPI;
namespace TFDLOP {
    namespace PriorBox {
        struct Priorboxparam {
            vector<float> minSizes, maxSizes, variance, aspect_ratio;
            float default_aspect_ratio = 1.0;
            bool clip, flip;
            float offset;
            float step_w, step_h;
            int numPriors;
            bool Reshaped = false;
        };

        void Prepare(TFContext tfContext, TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node),err);
            if(!err.empty()){
                printf("%s\n",err.c_str());
            }
            Priorboxparam *priorboxparam = new Priorboxparam();
            if(!param["priorBoxParam"].is_null()){
                priorboxparam->minSizes.clear();
                for (int i = 0; i < param["priorBoxParam"]["minSize"].array_items().size(); i++) {
                    priorboxparam->minSizes.push_back(param["priorBoxParam"]["minSize"].array_items()[i].number_value());
                }

                priorboxparam->maxSizes.clear();
                for (int i = 0; i < param["priorBoxParam"]["maxSize"].array_items().size(); i++) {
                    priorboxparam->maxSizes.push_back(param["priorBoxParam"]["maxSize"].array_items()[i].number_value());
                }

                priorboxparam->variance.clear();
                for (int i = 0; i < param["priorBoxParam"]["variance"].array_items().size(); i++) {
                    priorboxparam->variance.push_back(param["priorBoxParam"]["variance"].array_items()[i].number_value());
                }


                priorboxparam->default_aspect_ratio = (param["priorBoxParam"]["default_aspect_ratio"].is_null())?1.0:param["priorBoxParam"]["default_aspect_ratio"].number_value();
                priorboxparam->clip = param["priorBoxParam"]["clip"].bool_value();
                priorboxparam->flip = param["priorBoxParam"]["flip"].bool_value();
                if (!param["priorBoxParam"]["step"].is_null()) {
                    priorboxparam->step_h = param["priorBoxParam"]["step"].number_value();
                    priorboxparam->step_w = param["priorBoxParam"]["step"].number_value();
                } else if ((!param["priorBoxParam"]["step_h"].is_null()) &&
                           (!param["priorBoxParam"]["step_w"].is_null())) {
                    priorboxparam->step_h = param["priorBoxParam"]["step_h"].number_value();
                    priorboxparam->step_w = param["priorBoxParam"]["step_w"].number_value();
                } else {
                    priorboxparam->step_h = 0;
                    priorboxparam->step_w = 0;
                }


                priorboxparam->offset = (param["priorBoxParam"]["offset"].is_null())?0.5:param["priorBoxParam"]["offset"].number_value();
                std::vector<float> aspect_ratio_;
                for (int i = 0; i < param["priorBoxParam"]["aspectRatio"].array_items().size(); i++) {
                    aspect_ratio_.push_back(param["priorBoxParam"]["aspectRatio"].array_items()[i].number_value());
                }
                priorboxparam->aspect_ratio.clear();
                priorboxparam->aspect_ratio.push_back(priorboxparam->default_aspect_ratio);
                for (int i = 0; i < aspect_ratio_.size(); ++i) {
                    float ar = aspect_ratio_[i];
                    bool already_exist = false;
                    for (int j = 0; j < priorboxparam->aspect_ratio.size(); ++j) {
                        if (fabs(ar - priorboxparam->aspect_ratio[j]) < 1e-6) {
                            already_exist = true;
                            break;
                        }
                    }
                    if (!already_exist) {
                        priorboxparam->aspect_ratio.push_back(ar);
                        if (priorboxparam->flip) {
                            priorboxparam->aspect_ratio.push_back(1. / ar);
                        }
                    }
                }
                if (priorboxparam->aspect_ratio.empty()) {
                    priorboxparam->numPriors = priorboxparam->maxSizes.size() + priorboxparam->minSizes.size();
                } else {
                    priorboxparam->numPriors = priorboxparam->maxSizes.size() + priorboxparam->minSizes.size() *
                                                                                priorboxparam->aspect_ratio.size();
                }
            }
            else{
                priorboxparam->minSizes.clear();
                for (int i = 0; i < param["minSizes"].array_items().size(); i++) {
                    priorboxparam->minSizes.push_back(param["minSizes"].array_items()[i].number_value());
                }

                priorboxparam->maxSizes.clear();
                for (int i = 0; i < param["maxSizes"].array_items().size(); i++) {
                    priorboxparam->maxSizes.push_back(param["maxSizes"].array_items()[i].number_value());
                }

                priorboxparam->variance.clear();
                for (int i = 0; i < param["variance"].array_items().size(); i++) {
                    priorboxparam->variance.push_back(param["variance"].array_items()[i].number_value());
                }


                priorboxparam->default_aspect_ratio = (param["default_aspect_ratio"].is_null())?1.0:param["default_aspect_ratio"].number_value();
                priorboxparam->clip = param["clip"].bool_value();
                priorboxparam->flip = param["flip"].bool_value();
                if (!param["step"].is_null()) {
                    priorboxparam->step_h = param["step"].number_value();
                    priorboxparam->step_w = param["step"].number_value();
                } else if ((!param["step_h"].is_null()) &&
                           (!param["step_w"].is_null())) {
                    priorboxparam->step_h = param["step_h"].number_value();
                    priorboxparam->step_w = param["step_w"].number_value();
                } else {
                    priorboxparam->step_h = 0;
                    priorboxparam->step_w = 0;
                }


                priorboxparam->offset = (param["offset"].is_null())?0.5:param["offset"].number_value();
                std::vector<float> aspect_ratio_;
                for (int i = 0; i < param["aspect_ratio_"].array_items().size(); i++) {
                    aspect_ratio_.push_back(param["aspect_ratio_"].array_items()[i].number_value());
                }
                priorboxparam->aspect_ratio.clear();
                priorboxparam->aspect_ratio.push_back(priorboxparam->default_aspect_ratio);
                for (int i = 0; i < aspect_ratio_.size(); ++i) {
                    float ar = aspect_ratio_[i];
                    bool already_exist = false;
                    for (int j = 0; j < priorboxparam->aspect_ratio.size(); ++j) {
                        if (fabs(ar - priorboxparam->aspect_ratio[j]) < 1e-6) {
                            already_exist = true;
                            break;
                        }
                    }
                    if (!already_exist) {
                        priorboxparam->aspect_ratio.push_back(ar);
                        if (priorboxparam->flip) {
                            priorboxparam->aspect_ratio.push_back(1. / ar);
                        }
                    }
                }
                if (priorboxparam->aspect_ratio.empty()) {
                    priorboxparam->numPriors = priorboxparam->maxSizes.size() + priorboxparam->minSizes.size();
                } else {
                    priorboxparam->numPriors = priorboxparam->maxSizes.size() + priorboxparam->minSizes.size() *
                                                                                priorboxparam->aspect_ratio.size();
                }
            }

            // Get the Customparam ptr, default SDK will offer a empty ptr.
            //And SDK won't use this ptr ,so the ptr allocating or Free Depend on the User.
            FreeNodeCustomParam(node, [](void *custromparam) {
                delete (Priorboxparam *) custromparam;
            });
            NewNodeCustomParam(node,[&priorboxparam]()->void*{
                return priorboxparam;
            });


        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *custromparam) {
                delete (Priorboxparam *) custromparam;
            });
        }

        void Reshape(TFContext tfContext, TFNode node) {
            Priorboxparam *priorboxparam = (Priorboxparam *) GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            TFCHECK_EQ(info.InputNames.size(), 2);

            auto input_box = GetTensorByName(tfContext, info.InputNames[0]);
            auto input_img = GetTensorByName(tfContext, info.InputNames[1]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);
            vector<int> dims;
            dims.push_back(1);
            dims.push_back(2);
            dims.push_back(GetTensorCount(input_box, 2) * priorboxparam->numPriors * 4);
            ReSizeTensor(outputData, dims);
            SetTensorType(outputData, TFCAPI_FLOAT);
            priorboxparam->Reshaped = true;
        }

        void Eval(TFContext tfContext, TFNode node) {
            Priorboxparam *priorboxparam = (Priorboxparam *) GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            auto input_box = GetTensorByName(tfContext, info.InputNames[0]);
            auto input_img = GetTensorByName(tfContext, info.InputNames[1]);
            auto outputData = GetTensorByName(tfContext, info.OutputNames[0]);
            const int layerHeight = GetTensorDim(input_box, 2);
            const int layerWidth = GetTensorDim(input_box, 3);
            int imgHeight = GetTensorDim(input_img, 2);
            int imgWidth = GetTensorDim(input_img, 3);
            float *output = (float *) GetTensordata(outputData);
            int dim = layerHeight * layerWidth * priorboxparam->numPriors * 4;
            int idx = 0;
            if (!priorboxparam->Reshaped)return;

            priorboxparam->Reshaped = false;
            int step_w, step_h;
            if (priorboxparam->step_h == 0 && priorboxparam->step_w == 0) {
                step_w = static_cast<float>(imgWidth) / layerWidth;
                step_h = static_cast<float>(imgHeight) / layerHeight;
            } else {
                step_w = priorboxparam->step_w;
                step_h = priorboxparam->step_h;
            }
            for (int h = 0; h < layerHeight; ++h) {
                for (int w = 0; w < layerWidth; ++w) {
                    float centerX = (w + priorboxparam->offset) * step_w;
                    float centerY = (h + priorboxparam->offset) * step_h;
                    float boxWidth, boxHeight;
                    for (int s = 0; s < priorboxparam->minSizes.size(); ++s) {
                        float minSize = priorboxparam->minSizes[s];
                        // first prior: aspect_ratio = default_aspect_ratio (default=1.0), size = min_size
                        boxWidth = minSize * sqrt(priorboxparam->default_aspect_ratio);
                        boxHeight = minSize / sqrt(priorboxparam->default_aspect_ratio);
                        // xmin
                        output[idx++] = (centerX - boxWidth / 2.) / imgWidth;
                        // ymin
                        output[idx++] = (centerY - boxHeight / 2.) / imgHeight;
                        // xmax
                        output[idx++] = (centerX + boxWidth / 2.) / imgWidth;
                        // ymax
                        output[idx++] = (centerY + boxHeight / 2.) / imgHeight;

                        if (priorboxparam->maxSizes.size() > s) {
                            float maxSize = priorboxparam->maxSizes[s];
                            // second prior: aspect_ratio = default_aspect_ratio (default=1.0), size = sqrt(min_size * max_size)
                            boxWidth =
                                    sqrt((float) (minSize * maxSize)) * sqrt(priorboxparam->default_aspect_ratio);
                            boxHeight =
                                    sqrt((float) (minSize * maxSize)) / sqrt(priorboxparam->default_aspect_ratio);
                            // xmin
                            output[idx++] = (centerX - boxWidth / 2.) / imgWidth;
                            // ymin
                            output[idx++] = (centerY - boxHeight / 2.) / imgHeight;
                            // xmax
                            output[idx++] = (centerX + boxWidth / 2.) / imgWidth;
                            // ymax
                            output[idx++] = (centerY + boxHeight / 2.) / imgHeight;
                        }
                        for (int r = 0; r < priorboxparam->aspect_ratio.size(); ++r) {
                            float ar = priorboxparam->aspect_ratio[r];
                            if (fabs(ar - priorboxparam->default_aspect_ratio) < 1e-6) {
                                continue;
                            }
                            boxWidth = minSize * sqrt(ar);
                            boxHeight = minSize / sqrt(ar);
                            // xmin
                            output[idx++] = (centerX - boxWidth / 2.) / imgWidth;
                            // ymin
                            output[idx++] = (centerY - boxHeight / 2.) / imgHeight;
                            // xmax
                            output[idx++] = (centerX + boxWidth / 2.) / imgWidth;
                            // ymax
                            output[idx++] = (centerY + boxHeight / 2.) / imgHeight;
                        }
                    }
                }
            }

            // clip the prior's coordidate such that it is within [0, 1]
            if (priorboxparam->clip) {
                for (int d = 0; d < dim; ++d) {
                    output[d] = std::min(std::max(output[d], 0.f), 1.f);
                }
            }
            // set the variance.

            output += GetTensorCount(outputData, 2);
            int count = 0;
            for (int h = 0; h < layerHeight; ++h) {
                for (int w = 0; w < layerWidth; ++w) {
                    for (int i = 0; i < priorboxparam->numPriors; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            output[count++] = priorboxparam->variance[j];
                        }
                    }
                }
            }

        }


    }
    RegistOp(PriorBox)
    .Set(PriorBox::Prepare,PriorBox::Reshape,PriorBox::Eval,PriorBox::Free);

}


#endif //TFDL2_PRIORBOX_H

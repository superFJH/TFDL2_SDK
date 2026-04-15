//
// Created by test on 2019/10/14.
//

#ifndef TFDL2_DETECTIONOUTPUTS_H
#define TFDL2_DETECTIONOUTPUTS_H
#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include "bbox_util.h"
#include <iostream>
using namespace TFDL_CAPI;
namespace TFDLOP {
    namespace DetectionOutput {
        struct DetectOutputParam {
            bool mbox_reverse_; //special for mv00, need swap(mbox[0], mbox[1]), swap(mbox[2], mbox[3])

            int num_classes_;
            bool share_location_;
            int num_loc_classes_;
            int background_label_id_;
            TFDLBOX::CodeType code_type_ = TFDLBOX::CodeType::PriorBoxParameter_CodeType_CENTER_SIZE;
            bool variance_encoded_in_target_ = false;
            int keep_top_k_;
            float confidence_threshold_;
            bool clip_bbox;
            int num_;
            int num_priors_;

            float nms_threshold_;
            int top_k_;
            float eta_;
            // Retrieve all prior bboxes. It is same within a batch since we assume all
            // images in a batch are of same dimension.
            vector<TFDLBOX::NormalizedBBox> prior_bboxes;
            vector<vector<float> > prior_variances;
            bool Reshaped = false;
        };

        void Prepare(TFContext tfContext, TFNode node) {
            auto custromparam = GetNodeCustomParam(node);
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node),err);
            if(!err.empty()){
                printf("%s\n",err.c_str());
            }
            DetectOutputParam *detectOutputParam = new DetectOutputParam();
            if(!param["detectionOutputParam"].is_null()){
                detectOutputParam->num_classes_ = param["detectionOutputParam"]["numClasses"].int_value();
                detectOutputParam->share_location_ = param["detectionOutputParam"]["shareLocation"].bool_value();
                detectOutputParam->num_loc_classes_ = detectOutputParam->share_location_ ? 1
                                                                                         : detectOutputParam->num_classes_;
                detectOutputParam->background_label_id_ = param["detectionOutputParam"]["backgroundLabelId"].int_value();
                string codetype = param["detectionOutputParam"]["codeType"].string_value();
                if(codetype == "CENTER_SIZE"){
                    detectOutputParam->code_type_ =   TFDLBOX::CodeType::PriorBoxParameter_CodeType_CENTER_SIZE;
                }else if(codetype == "CORNER"){
                    detectOutputParam->code_type_ =   TFDLBOX::CodeType::PriorBoxParameter_CodeType_CORNER;
                }else if(codetype == "CORNER_SIZE"){
                    detectOutputParam->code_type_ =   TFDLBOX::CodeType::PriorBoxParameter_CodeType_CORNER_SIZE;
                }

                detectOutputParam->variance_encoded_in_target_ = param["detectionOutputParam"]["varianceEncodedInTarget"].bool_value();
                detectOutputParam->keep_top_k_ = param["detectionOutputParam"]["keepTopK"].is_null()?-1:param["detectionOutputParam"]["keepTopK"].int_value();
                detectOutputParam->confidence_threshold_ = param["detectionOutputParam"]["confidenceThreshold"].number_value();
                // Parameters used in nms.
                detectOutputParam->nms_threshold_ = param["detectionOutputParam"]["nmsParam"]["nmsThreshold"].number_value();
                TFCHECK_GE(detectOutputParam->nms_threshold_, 0.);
                detectOutputParam->eta_ = 1.0;
                detectOutputParam->top_k_ = param["detectionOutputParam"]["nmsParam"]["topK"].is_null()?-1:param["detectionOutputParam"]["nmsParam"]["topK"].int_value();
                detectOutputParam->mbox_reverse_ = false;//layerParameter->detection_output_param().switch_channels();
                detectOutputParam->clip_bbox = param["detectionOutputParam"]["clipBbox"].bool_value();
            }
            else{
                detectOutputParam->num_classes_ = param["num_classes_"].int_value();
                detectOutputParam->share_location_ = param["share_location_"].bool_value();
                detectOutputParam->num_loc_classes_ = detectOutputParam->share_location_ ? 1
                                                                                         : detectOutputParam->num_classes_;
                detectOutputParam->background_label_id_ = param["background_label_id_"].int_value();
                string codetype = param["code_type_"].string_value();
                if(codetype == "CENTER_SIZE"){
                    detectOutputParam->code_type_ =   TFDLBOX::CodeType::PriorBoxParameter_CodeType_CENTER_SIZE;
                }else if(codetype == "CORNER"){
                    detectOutputParam->code_type_ =   TFDLBOX::CodeType::PriorBoxParameter_CodeType_CORNER;
                }else if(codetype == "CORNER_SIZE"){
                    detectOutputParam->code_type_ =   TFDLBOX::CodeType::PriorBoxParameter_CodeType_CORNER_SIZE;
                }

                detectOutputParam->variance_encoded_in_target_ = param["variance_encoded_in_target_"].bool_value();
                detectOutputParam->keep_top_k_ = param["keep_top_k_"].is_null()?-1:param["keep_top_k_"].int_value();
                detectOutputParam->confidence_threshold_ = param["confidence_threshold_"].number_value();
                // Parameters used in nms.
                detectOutputParam->nms_threshold_ = param["nms_threshold_"].number_value();
                TFCHECK_GE(detectOutputParam->nms_threshold_, 0.);
                detectOutputParam->eta_ = 1.0;
                detectOutputParam->top_k_ = param["top_k_"].is_null()?-1:param["top_k_"].int_value();
                detectOutputParam->mbox_reverse_ = false;//layerParameter->detection_output_param().switch_channels();
                detectOutputParam->clip_bbox = param["clip_bbox"].bool_value();
            }


            FreeNodeCustomParam(node, [](void *custromparam) {
                delete (DetectOutputParam *) custromparam;
            });
            NewNodeCustomParam(node,[&detectOutputParam]()->void*{
                return detectOutputParam;
            });


        }

        void Reshape(TFContext tfContext, TFNode node) {
            DetectOutputParam *detectOutputParam = (DetectOutputParam *) GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            TFCHECK_EQ(info.InputNames.size(), 3);
            auto locData = GetTensorByName(tfContext,info.InputNames[0]);
            auto configData = GetTensorByName(tfContext,info.InputNames[1]);
            auto priorboxData = GetTensorByName(tfContext,info.InputNames[2]);
            auto outputData = GetTensorByName(tfContext,info.OutputNames[0]);

            TFCHECK_EQ(GetTensorType(priorboxData), TFCAPI_FLOAT);

            detectOutputParam->num_priors_ = GetTensorShape(priorboxData)[2] / 4;
            TFCHECK_EQ(detectOutputParam->num_priors_ * detectOutputParam->num_loc_classes_ * 4,
                       GetTensorShape(locData)[1]);
            TFCHECK_EQ(detectOutputParam->num_priors_ * detectOutputParam->num_classes_, GetTensorShape(configData)[1]);
            // num() and channels() are 1.
            vector<int> top_shape(2, 1);
            // Since the number of bboxes to be kept is unknown before nms, we manually
            // set it to (fake) 1.
            top_shape.push_back(1);
            // Each row is a 7 dimension vector, which stores
            // [image_id, label, confidence, xmin, ymin, xmax, ymax]
            top_shape.push_back(7);
            ReSizeTensor(outputData,top_shape);
            SetTensorType(outputData,TFCAPI_FLOAT);
            detectOutputParam->Reshaped = true;
        }

        void Eval(TFContext tfContext, TFNode node) {
            DetectOutputParam *detectOutputParam = (DetectOutputParam *) GetNodeCustomParam(node);
            auto info = GetNodeInfo(node);
            auto locData = GetTensorByName(tfContext,info.InputNames[0]);
            auto configData = GetTensorByName(tfContext,info.InputNames[1]);
            auto priorboxData = GetTensorByName(tfContext,info.InputNames[2]);
            auto outputData = GetTensorByName(tfContext,info.OutputNames[0]);
            float *loc_data;
            float *conf_data;
            if (detectOutputParam->Reshaped) {
                float *prior_data = (float*)GetTensordata(priorboxData);

                GetPriorBBoxes(prior_data, detectOutputParam->num_priors_, &detectOutputParam->prior_bboxes,
                               &detectOutputParam->prior_variances);
                detectOutputParam->Reshaped = false;
            }
            if (GetTensorType(locData) == TFCAPI_UINT8) {
                int cnt = GetTensorCount(locData,0);
                auto locconfig = GetTensorQuantizeInfo(tfContext,GetTensorName(locData));
                loc_data = new float[cnt];
                DeQuantizeTensorData(loc_data,(uint8_t*)GetTensordata(locData),cnt,locconfig);
            } else {
                loc_data = (float*)GetTensordata(locData);
            }
            if (GetTensorType(configData) == TFCAPI_UINT8) {
                int cnt = GetTensorCount(configData,0);
                auto conf_config = GetTensorQuantizeInfo(tfContext,GetTensorName(configData));
                conf_data = new float[cnt];
                DeQuantizeTensorData(conf_data,(uint8_t*)GetTensordata(configData),cnt,conf_config);
            } else {
                conf_data = (float*)GetTensordata(configData);
            }

            const int num = GetTensorShape(locData)[0];

            // Retrieve all location predictions.
            vector<TFDLBOX::LabelBBox> all_loc_preds;

            // Retrieve all confidences.
            vector<map<int, vector<float> > > all_conf_scores;

            // Decode all loc predictions to bboxes.
            vector<TFDLBOX::LabelBBox> all_decode_bboxes;
            if (detectOutputParam->mbox_reverse_) {
                int cnt = GetTensorCount(locData,0);
                for (int i = 0; i < cnt; i += 2) {
                    std::swap(loc_data[i], loc_data[i + 1]);
                }
            }
            TFDLBOX::GetLocPredictions(loc_data, num, detectOutputParam->num_priors_,
                                       detectOutputParam->num_loc_classes_,
                                       detectOutputParam->share_location_, &all_loc_preds);
            TFDLBOX::GetConfidenceScores(conf_data, num, detectOutputParam->num_priors_,
                                         detectOutputParam->num_classes_,
                                         &all_conf_scores);
            TFDLBOX::DecodeBBoxesAll(all_loc_preds, detectOutputParam->prior_bboxes,
                                     detectOutputParam->prior_variances, num,
                                     detectOutputParam->share_location_, detectOutputParam->num_loc_classes_,
                                     detectOutputParam->background_label_id_,
                                     detectOutputParam->code_type_, detectOutputParam->variance_encoded_in_target_,
                                     detectOutputParam->clip_bbox,
                                     &all_decode_bboxes);

            int num_kept = 0;
            vector<map<int, vector<int> > > all_indices;

            for (int i = 0; i < num; ++i) {
                const TFDLBOX::LabelBBox &decode_bboxes = all_decode_bboxes[i];
                const map<int, vector<float> > &conf_scores = all_conf_scores[i];
                map<int, vector<int> > indices;
                int num_det = 0;

                for (int c = 0; c < detectOutputParam->num_classes_; ++c) {
                    if (c == detectOutputParam->background_label_id_) {
                        // Ignore background class.
                        continue;
                    }
                    if (conf_scores.find(c) == conf_scores.end()) {
                        // Something bad happened if there are no predictions for current label.
                        //throw_exception("Could not find confidence predictions for label", new TFDLInitException());
                    }
                    const vector<float> &scores = conf_scores.find(c)->second;
                    int label = detectOutputParam->share_location_ ? -1 : c;
                    if (decode_bboxes.find(label) == decode_bboxes.end()) {
                        // Something bad happened if there are no predictions for current label.
                        //throw_exception("Could not find location predictions for label ", new TFDLInitException());
                        continue;
                    }
                    const vector<TFDLBOX::NormalizedBBox> &bboxes = decode_bboxes.find(label)->second;
                    TFDLBOX::ApplyNMSFast(bboxes, scores, detectOutputParam->confidence_threshold_,
                                          detectOutputParam->nms_threshold_, detectOutputParam->eta_,
                                          detectOutputParam->top_k_, &(indices[c]));
                    num_det += indices[c].size();
                }
                if (detectOutputParam->keep_top_k_ > -1 && num_det > detectOutputParam->keep_top_k_) {
                    vector<pair<float, pair<int, int> > > score_index_pairs;
                    for (map<int, vector<int> >::iterator it = indices.begin();
                         it != indices.end(); ++it) {
                        int label = it->first;
                        const vector<int> &label_indices = it->second;
                        if (conf_scores.find(label) == conf_scores.end()) {
                            // Something bad happened for current label.
                            //throw_exception("Could not find location predictions", new TFDLInitException());
                            continue;
                        }
                        const vector<float> &scores = conf_scores.find(label)->second;
                        for (int j = 0; j < label_indices.size(); ++j) {
                            int idx = label_indices[j];
                            TFCHECK_LT(idx, scores.size());
                            score_index_pairs.push_back(std::make_pair(
                                    scores[idx], std::make_pair(label, idx)));
                        }
                    }
                    // Keep top k results per image.
                    std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                              TFDLBOX::SortScorePairDescend<pair<int, int> >);
                    score_index_pairs.resize(detectOutputParam->keep_top_k_);
                    // Store the new indices.
                    map<int, vector<int> > new_indices;
                    for (int j = 0; j < score_index_pairs.size(); ++j) {
                        int label = score_index_pairs[j].second.first;
                        int idx = score_index_pairs[j].second.second;
                        new_indices[label].push_back(idx);
                    }
                    all_indices.push_back(new_indices);
                    num_kept += detectOutputParam->keep_top_k_;
                } else {
                    all_indices.push_back(indices);
                    num_kept += num_det;
                }
            }


            vector<int> top_shape(2, 1);
            top_shape.push_back(num_kept);
            top_shape.push_back(7);
            float *top_data;
            if (num_kept == 0) {
                //tlog->TFLOG("Couldn't find any detections");
                top_shape[2] = num;
                ReSizeTensor(outputData,top_shape);
                AllocateCpuMem(outputData);
                top_data = (float*)GetTensordata(outputData);
                std::fill(top_data, top_data + GetTensorCount(outputData,0), -1);
                // Generate fake results per image.
                for (int i = 0; i < num; ++i) {
                    top_data[0] = -1;
                    top_data += 7;
                }
            } else {
                ReSizeTensor(outputData,top_shape);
                AllocateCpuMem(outputData);
                top_data = (float*)GetTensordata(outputData);
            }


            int count = 0;
            for (int i = 0; i < num; ++i) {
                const map<int, vector<float> > &conf_scores = all_conf_scores[i];
                const TFDLBOX::LabelBBox &decode_bboxes = all_decode_bboxes[i];
                for (map<int, vector<int> >::iterator it = all_indices[i].begin();
                     it != all_indices[i].end(); ++it) {
                    int label = it->first;
                    if (conf_scores.find(label) == conf_scores.end()) {
                        // Something bad happened if there are no predictions for current label.
                        std::cout << "Could not find confidence predictions for " << label;
                        continue;
                    }
                    const vector<float> &scores = conf_scores.find(label)->second;
                    int loc_label = detectOutputParam->share_location_ ? -1 : label;
                    if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
                        // Something bad happened if there are no predictions for current label.
                        std::cout << "Could not find location predictions for " << loc_label << std::endl;
                        continue;
                    }
                    const vector<TFDLBOX::NormalizedBBox> &bboxes =
                            decode_bboxes.find(loc_label)->second;
                    vector<int> &indices = it->second;
                    for (int j = 0; j < indices.size(); ++j) {
                        int idx = indices[j];
                        top_data[count * 7] = i;
                        top_data[count * 7 + 1] = label;
                        top_data[count * 7 + 2] = scores[idx];
                        const TFDLBOX::NormalizedBBox &bbox = bboxes[idx];
                        top_data[count * 7 + 3] = bbox.xmin();
                        top_data[count * 7 + 4] = bbox.ymin();
                        top_data[count * 7 + 5] = bbox.xmax();
                        top_data[count * 7 + 6] = bbox.ymax();
                        ++count;
                    }

                }
            }
            if (GetTensorType(locData) == TFCAPI_UINT8) {
                delete[] loc_data;
            }

            if (GetTensorType(configData) == TFCAPI_UINT8) {
                delete[] conf_data;
            }
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *custromparam) {
                delete (DetectOutputParam *) custromparam;
            });
        }
    }
    RegistOp(DetectionOutput)
    .Set(DetectionOutput::Prepare,DetectionOutput::Reshape,DetectionOutput::Eval,DetectionOutput::Free);
}

#endif //TFDL2_DETECTIONOUTPUTS_H

//
// LocateAnythingEmbedMerge
//
// Converts token ids to embeddings and replaces/expands image placeholder
// positions with projected image features.
//
// Inputs:
//   [0] input_ids:      [1, raw_seq] int32/int64/float
//   [1] image_features: [1, image_tokens, hidden] or [image_tokens, hidden]
//   [2] embed_tokens:   [vocab, hidden] float param/tensor
//
// Output:
//   [0] hidden: [1, output_seq, hidden] float
//
// JSON:
//   "imageTokenId": int
//   "outputSeqLen": int, optional. If omitted, infer replacement length.
//

#ifndef NPU40T_LOCATE_ANYTHING_EMBED_MERGE_H
#define NPU40T_LOCATE_ANYTHING_EMBED_MERGE_H

#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

using namespace TFDL_CAPI;

namespace TFDLOP {
    namespace LocateAnythingEmbedMerge {
        struct Param {
            int imageTokenId = -1;
            int outputSeqLen = 0;
        };

        static Param parseParam(TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node), err);
            if (!err.empty()) {
                printf("[LocateAnythingEmbedMerge] JSON parse error: %s\n", err.c_str());
            }
            Param p;
            p.imageTokenId = param["imageTokenId"].int_value();
            p.outputSeqLen = param["outputSeqLen"].int_value();
            return p;
        }

        static TFTensor getTensorOrParam(TFContext tfContext, const string &name) {
            auto tensor = GetTensorByName(tfContext, name);
            if (!tensor.IsValid()) {
                tensor = GetParam(tfContext, name);
            }
            return tensor;
        }

        void Prepare(TFContext tfContext, TFNode node) {
            auto *p = new Param(parseParam(node));

            FreeNodeCustomParam(node, [](void *cp) { delete (Param *)cp; });
            NewNodeCustomParam(node, [&p]() -> void * { return p; });
        }

        static int64_t readId(const void *ptr, int index, TFCAPI_DATATYPE type) {
            if (type == TFCAPI_INT64) {
                return ((const int64_t *)ptr)[index];
            }
            if (type == TFCAPI_INT32) {
                return ((const int32_t *)ptr)[index];
            }
            if (type == TFCAPI_FLOAT) {
                return (int64_t)std::llround(((const float *)ptr)[index]);
            }
            if (type == TFCAPI_UINT8) {
                return (int64_t)((const uint8_t *)ptr)[index];
            }
            return 0;
        }

        static int inferOutputSeq(
            TFContext tfContext,
            TFNode node,
            Param *p,
            const std::vector<int> &idsShape,
            const std::vector<int> &imageShape
        ) {
            if (p->outputSeqLen > 0) {
                return p->outputSeqLen;
            }

            int rawSeq = idsShape[idsShape.size() - 1];
            int imageTokens = (imageShape.size() == 3) ? imageShape[1] : imageShape[0];
            auto info = GetNodeInfo(node);
            auto ids = GetTensorByName(tfContext, info.InputNames[0]);
            const void *idsPtr = GetTensordata(ids);
            auto idsType = GetTensorType(ids);
            int placeholders = 0;
            for (int i = 0; i < rawSeq; i++) {
                if ((int)readId(idsPtr, i, idsType) == p->imageTokenId) {
                    placeholders++;
                }
            }
            if (placeholders == 1) {
                return rawSeq - 1 + imageTokens;
            }
            return rawSeq;
        }

        void Reshape(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            TFCHECK_EQ(info.InputNames.size(), 3);
            TFCHECK_EQ(info.OutputNames.size(), 1);

            Param fallback;
            Param *p = (Param *)GetNodeCustomParam(node);
            if (p == nullptr) {
                fallback = parseParam(node);
                p = &fallback;
            }
            auto ids = GetTensorByName(tfContext, info.InputNames[0]);
            auto image = GetTensorByName(tfContext, info.InputNames[1]);
            auto embed = getTensorOrParam(tfContext, info.InputNames[2]);
            auto out = GetTensorByName(tfContext, info.OutputNames[0]);

            if (!ids.IsValid() || !image.IsValid() || !embed.IsValid() || !out.IsValid()) {
                fprintf(
                    stderr,
                    "[LocateAnythingEmbedMerge] invalid tensor: ids=%d image=%d embed=%d out=%d\n",
                    ids.IsValid(), image.IsValid(), embed.IsValid(), out.IsValid()
                );
                fflush(stderr);
                return;
            }

            auto idsShape = GetTensorShape(ids);
            auto imageShape = GetTensorShape(image);
            auto embedShape = GetTensorShape(embed);
            TFCHECK_EQ(idsShape.size(), 2);
            TFCHECK_GE(imageShape.size(), 2);
            TFCHECK_EQ(embedShape.size(), 2);
            int hidden = embedShape[1];
            int imageHidden = imageShape[imageShape.size() - 1];
            TFCHECK_EQ(hidden, imageHidden);

            int outputSeq = inferOutputSeq(tfContext, node, p, idsShape, imageShape);
            ReSizeTensor(out, {idsShape[0], outputSeq, hidden});
            SetTensorType(out, TFCAPI_FLOAT);
        }

        void Eval(TFContext tfContext, TFNode node) {
            Param fallback;
            Param *p = (Param *)GetNodeCustomParam(node);
            if (p == nullptr) {
                fallback = parseParam(node);
                p = &fallback;
            }
            auto info = GetNodeInfo(node);

            auto ids = GetTensorByName(tfContext, info.InputNames[0]);
            auto image = GetTensorByName(tfContext, info.InputNames[1]);
            auto embed = getTensorOrParam(tfContext, info.InputNames[2]);
            auto out = GetTensorByName(tfContext, info.OutputNames[0]);

            auto idsShape = GetTensorShape(ids);
            auto imageShape = GetTensorShape(image);
            auto embedShape = GetTensorShape(embed);
            int rawSeq = idsShape[1];
            int vocab = embedShape[0];
            int hidden = embedShape[1];
            int imageTokens = (imageShape.size() == 3) ? imageShape[1] : imageShape[0];

            const void *idsPtr = GetTensordata(ids);
            auto idsType = GetTensorType(ids);
            const float *embedPtr = (const float *)GetTensordata(embed);

            std::vector<float> imageFloat(imageTokens * hidden);
            const float *imagePtr = nullptr;
            if (GetTensorType(image) == TFCAPI_FLOAT) {
                imagePtr = (const float *)GetTensordata(image);
            } else if (GetTensorType(image) == TFCAPI_UINT8) {
                auto imageQ = GetTensorQuantizeInfo(tfContext, info.InputNames[1]);
                DeQuantizeTensorData(
                    imageFloat.data(),
                    (uint8_t *)GetTensordata(image),
                    imageTokens * hidden,
                    imageQ
                );
                imagePtr = imageFloat.data();
            } else {
                printf("[LocateAnythingEmbedMerge] unsupported image feature dtype\n");
                return;
            }

            float *outPtr = (float *)GetTensordata(out);
            auto outShape = GetTensorShape(out);
            int outputSeq = outShape[1];
            memset(outPtr, 0, outputSeq * hidden * sizeof(float));

            int outPos = 0;
            int imageCursor = 0;
            int placeholders = 0;
            for (int i = 0; i < rawSeq; i++) {
                if ((int)readId(idsPtr, i, idsType) == p->imageTokenId) {
                    placeholders++;
                }
            }
            bool expandSinglePlaceholder = placeholders == 1 && imageTokens != 1;

            for (int i = 0; i < rawSeq; i++) {
                if (outPos >= outputSeq) {
                    break;
                }
                int64_t tokenId = readId(idsPtr, i, idsType);
                if ((int)tokenId == p->imageTokenId) {
                    if (expandSinglePlaceholder) {
                        int copyTokens = std::min(imageTokens, outputSeq - outPos);
                        memcpy(
                            outPtr + outPos * hidden,
                            imagePtr,
                            copyTokens * hidden * sizeof(float)
                        );
                        outPos += copyTokens;
                    } else {
                        int imageIndex = std::min(imageCursor, imageTokens - 1);
                        memcpy(
                            outPtr + outPos * hidden,
                            imagePtr + imageIndex * hidden,
                            hidden * sizeof(float)
                        );
                        imageCursor++;
                        outPos++;
                    }
                    continue;
                }

                if (tokenId < 0 || tokenId >= vocab) {
                    memset(outPtr + outPos * hidden, 0, hidden * sizeof(float));
                } else {
                    memcpy(
                        outPtr + outPos * hidden,
                        embedPtr + tokenId * hidden,
                        hidden * sizeof(float)
                    );
                }
                outPos++;
            }
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *cp) { delete (Param *)cp; });
        }
    }

    RegistOp(LocateAnythingEmbedMerge)
    .Set(LocateAnythingEmbedMerge::Prepare,
         LocateAnythingEmbedMerge::Reshape,
         LocateAnythingEmbedMerge::Eval,
         LocateAnythingEmbedMerge::Free);
}

#endif // NPU40T_LOCATE_ANYTHING_EMBED_MERGE_H

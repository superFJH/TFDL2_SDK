//
// Convert DA3 camera decoder pose encoding [t(3), q(4), fov(2)]
// into extrinsics (world-to-camera, 3x4) and intrinsics (3x3).
//

#ifndef NPU40T_POSE_ENCODING_TO_CAMERA_H
#define NPU40T_POSE_ENCODING_TO_CAMERA_H

#include "TFDL2_C_API.h"
#include "CustomCommon.h"
#include "json11.hpp"
#include <cassert>
#include <cmath>

using namespace TFDL_CAPI;
namespace TFDLOP {
    namespace PoseEncodingToCamera {
        struct PoseEncodingToCameraParam {
            int imageH = 518;
            int imageW = 518;
        };

        void Prepare(TFContext tfContext, TFNode node) {
            json11::Json param;
            string err;
            param = json11::Json::parse(GetNodeCustomJsonStr(node), err);

            auto *p = new PoseEncodingToCameraParam();
            if (err.empty()) {
                auto hw = param["image_size_hw"].array_items();
                if (hw.size() == 2) {
                    p->imageH = hw[0].int_value();
                    p->imageW = hw[1].int_value();
                }
            }
            FreeNodeCustomParam(node, [](void *customparam) {
                delete (PoseEncodingToCameraParam *)customparam;
            });
            NewNodeCustomParam(node, [&p]() -> void * {
                return p;
            });
        }

        void Reshape(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            TFCHECK_EQ(info.InputNames.size(), 1);
            TFCHECK_EQ(info.OutputNames.size(), 2);
            auto poseData = GetTensorByName(tfContext, info.InputNames[0]);
            auto poseShape = GetTensorShape(poseData);
            TFCHECK_EQ(poseShape.size(), 3);
            TFCHECK_EQ(poseShape[2], 9);

            auto extriData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto intriData = GetTensorByName(tfContext, info.OutputNames[1]);

            ReSizeTensor(extriData, {poseShape[0], poseShape[1], 3, 4});
            SetTensorType(extriData, GetTensorType(poseData));
            ReSizeTensor(intriData, {poseShape[0], poseShape[1], 3, 3});
            SetTensorType(intriData, GetTensorType(poseData));
        }

        static inline void quatToMatrix(const float *q, float *R) {
            float qw = q[0];
            float qx = q[1];
            float qy = q[2];
            float qz = q[3];
            float norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz + 1e-12f);
            qw /= norm;
            qx /= norm;
            qy /= norm;
            qz /= norm;

            R[0] = 1.0f - 2.0f * (qy * qy + qz * qz);
            R[1] = 2.0f * (qx * qy - qz * qw);
            R[2] = 2.0f * (qx * qz + qy * qw);
            R[3] = 2.0f * (qx * qy + qz * qw);
            R[4] = 1.0f - 2.0f * (qx * qx + qz * qz);
            R[5] = 2.0f * (qy * qz - qx * qw);
            R[6] = 2.0f * (qx * qz - qy * qw);
            R[7] = 2.0f * (qy * qz + qx * qw);
            R[8] = 1.0f - 2.0f * (qx * qx + qy * qy);
        }

        void Eval(TFContext tfContext, TFNode node) {
            auto info = GetNodeInfo(node);
            auto *param = (PoseEncodingToCameraParam *)GetNodeCustomParam(node);
            auto poseData = GetTensorByName(tfContext, info.InputNames[0]);
            auto extriData = GetTensorByName(tfContext, info.OutputNames[0]);
            auto intriData = GetTensorByName(tfContext, info.OutputNames[1]);
            auto poseShape = GetTensorShape(poseData);
            int B = poseShape[0];
            int V = poseShape[1];

            const float *posePtr = (const float *)GetTensordata(poseData);
            float *extriPtr = (float *)GetTensordata(extriData);
            float *intriPtr = (float *)GetTensordata(intriData);

            for (int b = 0; b < B; b++) {
                for (int v = 0; v < V; v++) {
                    const float *p = posePtr + (b * V + v) * 9;
                    const float *t = p;
                    const float *q = p + 3;
                    const float *fov = p + 7;

                    float R[9];
                    quatToMatrix(q, R);

                    float Rt[9] = {
                        R[0], R[3], R[6],
                        R[1], R[4], R[7],
                        R[2], R[5], R[8],
                    };
                    float invT[3] = {
                        -(Rt[0] * t[0] + Rt[1] * t[1] + Rt[2] * t[2]),
                        -(Rt[3] * t[0] + Rt[4] * t[1] + Rt[5] * t[2]),
                        -(Rt[6] * t[0] + Rt[7] * t[1] + Rt[8] * t[2]),
                    };

                    float *ex = extriPtr + (b * V + v) * 12;
                    ex[0] = Rt[0]; ex[1] = Rt[1]; ex[2] = Rt[2]; ex[3] = invT[0];
                    ex[4] = Rt[3]; ex[5] = Rt[4]; ex[6] = Rt[5]; ex[7] = invT[1];
                    ex[8] = Rt[6]; ex[9] = Rt[7]; ex[10] = Rt[8]; ex[11] = invT[2];

                    float fx = 0.5f * (float)param->imageW / std::tan(0.5f * fov[0]);
                    float fy = 0.5f * (float)param->imageH / std::tan(0.5f * fov[1]);
                    float cx = 0.5f * (float)param->imageW;
                    float cy = 0.5f * (float)param->imageH;

                    float *in = intriPtr + (b * V + v) * 9;
                    in[0] = fx;  in[1] = 0.0f; in[2] = cx;
                    in[3] = 0.0f; in[4] = fy;  in[5] = cy;
                    in[6] = 0.0f; in[7] = 0.0f; in[8] = 1.0f;
                }
            }
        }

        void Free(TFContext tfContext, TFNode node) {
            FreeNodeCustomParam(node, [](void *customparam) {
                delete (PoseEncodingToCameraParam *)customparam;
            });
        }
    }
    RegistOp(PoseEncodingToCamera)
    .Set(
        PoseEncodingToCamera::Prepare,
        PoseEncodingToCamera::Reshape,
        PoseEncodingToCamera::Eval,
        PoseEncodingToCamera::Free
    );
}

#endif

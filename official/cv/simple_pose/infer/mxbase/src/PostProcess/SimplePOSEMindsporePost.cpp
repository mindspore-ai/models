/*
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "SimplePOSEMindsporePost.h"
#include <math.h>
#include <stdio.h>
#include<string>
#include<memory>
#include <boost/property_tree/json_parser.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "acl/acl.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"

namespace {
// Output Tensor
auto floatDeleter = [](float* p) {};
const int OUTPUT_TENSOR_SIZE = 1;
const int OUTPUT_BBOX_SIZE = 4;
const int OUTPUT_BBOX_INDEX = 0;
const int NPOINTS = 17;
float heatmaps_reshape[NPOINTS][3072] = {};
float heatmaps_reshape_ex[NPOINTS][3072] = {};
float batch_heatmaps[NPOINTS][64][48] = {};
float batch_heatmaps_ex[NPOINTS][64][48] = {};
}  // namespace

namespace MxBase {
SimplePOSEMindsporePost &SimplePOSEMindsporePost::operator=(const SimplePOSEMindsporePost &other) {
    if (this == &other) {
        return *this;
    }
    ObjectPostProcessBase::operator=(other);
    return *this;
}

APP_ERROR SimplePOSEMindsporePost::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
    LogInfo << "Begin to initialize SimplePOSEMindsporePost.";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superinit  in ObjectPostProcessBase.";
        return ret;
    }
    LogInfo << "End to initialize SimplePOSEMindsporePost.";
    return APP_ERR_OK;
}

APP_ERROR SimplePOSEMindsporePost::DeInit() {
    LogInfo << "Begin to deinitialize SimplePOSEMindsporePost.";
    LogInfo << "End to deinitialize SimplePOSEMindsporePost.";
    return APP_ERR_OK;
}

bool SimplePOSEMindsporePost::IsValidTensors(const std::vector<TensorBase> &tensors) const {
    if (tensors.size() < OUTPUT_TENSOR_SIZE) {
        LogError << "The number of tensor (" << tensors.size() << ") is less than required (" << OUTPUT_TENSOR_SIZE
                 << ")";
        return false;
    }
    auto bboxShape = tensors[OUTPUT_BBOX_INDEX].GetShape();
    if (bboxShape.size() != OUTPUT_BBOX_SIZE) {
        LogError << "The number of tensor[" << OUTPUT_BBOX_INDEX << "] dimensions (" << bboxShape.size()
                 << ") is not equal to (" << OUTPUT_BBOX_SIZE << ")";
        return false;
    }
    return true;
}
static void get_affine(const float center[], const float scale[], cv::Mat *warp_mat) {
    float scale_tem[2] = {};
    scale_tem[0] = scale[0] * 200;
    scale_tem[1] = scale[1] * 200;
    float src_w = scale_tem[0];
    float dst_w = 48;
    float dst_h = 64;
    float src_dir[2] = {};
    float dst_dir[2] = {};

    float sn = sin(0);
    float cs = cos(0);
    src_dir[0] = src_w * 0.5 * sn;
    src_dir[1] = src_w * (-0.5) * cs;
    dst_dir[0] = 0;
    dst_dir[1] = dst_w * (-0.5);

    float src[3][2] = {};
    float dst[3][2] = {};

    src[0][0] = center[0];
    src[0][1] = center[1];
    src[1][0] = center[0] + src_dir[0];
    src[1][1] = center[1] + src_dir[1];
    dst[0][0] = dst_w * 0.5;
    dst[0][1] = dst_h * 0.5;
    dst[1][0] = dst_w * 0.5 + dst_dir[0];
    dst[1][1] = dst_h * 0.5 + dst_dir[1];

    float src_direct[2] = {};
    src_direct[0] = src[0][0] - src[1][0];
    src_direct[1] = src[0][1] - src[1][1];
    src[2][0] = src[1][0] - src_direct[1];
    src[2][1] = src[1][1] + src_direct[0];

    float dst_direct[2] = {};
    dst_direct[0] = dst[0][0] - dst[1][0];
    dst_direct[1] = dst[0][1] - dst[1][1];
    dst[2][0] = dst[1][0] - dst_direct[1];
    dst[2][1] = dst[1][1] + dst_direct[0];
    cv::Point2f srcPoint2f[3], dstPoint2f[3];
    srcPoint2f[0] = cv::Point2f(static_cast<float>(src[0][0]), static_cast<float>(src[0][1]));
    srcPoint2f[1] = cv::Point2f(static_cast<float>(src[1][0]), static_cast<float>(src[1][1]));
    srcPoint2f[2] = cv::Point2f(static_cast<float>(src[2][0]), static_cast<float>(src[2][1]));
    dstPoint2f[0] = cv::Point2f(static_cast<float>(dst[0][0]), static_cast<float>(dst[0][1]));
    dstPoint2f[1] = cv::Point2f(static_cast<float>(dst[1][0]), static_cast<float>(dst[1][1]));
    dstPoint2f[2] = cv::Point2f(static_cast<float>(dst[2][0]), static_cast<float>(dst[2][1]));
    cv::Mat warp_mat_af(2, 3, CV_32FC1);
    warp_mat_af = cv::getAffineTransform(dstPoint2f, srcPoint2f);
    *warp_mat = warp_mat_af;
}

static double get_float_data(const int index, const float(*heatmaps_reshape)[3072]) {
    float tem = 0;
    for (int j = 0; j < 3072; j++) {
        if (heatmaps_reshape[index][j] > tem) {
            tem = heatmaps_reshape[index][j];
        }
    }
    return tem;
}
static int get_int_data(const int index, const float(*heatmaps_reshape)[3072]) {
    int idx_tem = 0;
    float tem = 0;
    for (int j = 0; j < 3072; j++) {
        if (heatmaps_reshape[index][j] > tem) {
            tem = heatmaps_reshape[index][j];
            idx_tem = j;
        }
    }
    return idx_tem;
}
static void get_data(const std::vector<TensorBase>& tensors,
    const std::vector<TensorBase>& tensors1,
    uint32_t heatmapHeight, uint32_t heatmapWeight) {
    auto bboxPtr = reinterpret_cast<float*>(tensors[OUTPUT_BBOX_INDEX].GetBuffer());
    auto bboxPtr1 = reinterpret_cast<float*>(tensors1[OUTPUT_BBOX_INDEX].GetBuffer());
    std::shared_ptr<void> keypoint_pointer;
    keypoint_pointer.reset(bboxPtr, floatDeleter);
    std::shared_ptr<void> keypoint_pointer_ex;
    keypoint_pointer_ex.reset(bboxPtr1, floatDeleter);

    for (size_t i = 0; i < NPOINTS; i++) {
        int startIndex = i * heatmapHeight * heatmapWeight;
        for (size_t j = 0; j < heatmapHeight; j++) {
            int middleIndex = j * heatmapWeight;
            for (size_t k = 0; k < heatmapWeight; k++) {
                float x = static_cast<float*>(keypoint_pointer.get())[startIndex + j * heatmapWeight + k];
                heatmaps_reshape[i][j * heatmapWeight + k] = x;
                batch_heatmaps[i][j][k] = x;
            }
        }
    }

    for (size_t j = 0; j < heatmapHeight; j++) {
        int middleIndex = j * heatmapWeight;
        batch_heatmaps_ex[0][j][0] = static_cast<float*>(keypoint_pointer_ex.get())[(j + 1) * heatmapWeight - 1];
        heatmaps_reshape_ex[0][j * heatmapWeight] =
            static_cast<float*>(keypoint_pointer_ex.get())[(j + 1) * heatmapWeight - 1];
        for (size_t k = 0; k < heatmapWeight - 1; k++) {
            float x = static_cast<float*>(keypoint_pointer_ex.get())[(j + 1) * heatmapWeight - k - 1];
            batch_heatmaps_ex[0][j][k + 1] = x;
            heatmaps_reshape_ex[0][j * heatmapWeight + k + 1] = x;
        }
    }

    for (size_t i = 1; i < NPOINTS; i += 2) {
        int startIndex0 = i * heatmapHeight * heatmapWeight;
        int startIndex1 = (i + 1) * heatmapHeight * heatmapWeight;
        for (size_t j = 0; j < heatmapHeight; j++) {
            int middleIndex = j * heatmapWeight;
            batch_heatmaps_ex[i][j][0] =
                static_cast<float*>(keypoint_pointer_ex.get())[startIndex1 + (j + 1) * heatmapWeight - 1];
            batch_heatmaps_ex[i + 1][j][0] =
                static_cast<float*>(keypoint_pointer_ex.get())[startIndex0 + (j + 1) * heatmapWeight - 1];
            heatmaps_reshape_ex[i][j * heatmapWeight + 1] =
                static_cast<float*>(keypoint_pointer_ex.get())[startIndex1 + (j + 1) * heatmapWeight - 1];
            heatmaps_reshape_ex[i + 1][j * heatmapWeight + 1] =
                static_cast<float*>(keypoint_pointer_ex.get())[startIndex0 + (j + 1) * heatmapWeight - 1];
            for (size_t k = 0; k < heatmapWeight - 1; k++) {
                float x0 =
                    static_cast<float*>(keypoint_pointer_ex.get())[startIndex0 + (j + 1) * heatmapWeight - k - 1];
                float x1 =
                    static_cast<float*>(keypoint_pointer_ex.get())[startIndex1 + (j + 1) * heatmapWeight - k - 1];
                batch_heatmaps_ex[i][j][k + 1] = x1;
                batch_heatmaps_ex[i + 1][j][k + 1] = x0;
                heatmaps_reshape_ex[i][j * heatmapWeight + k + 1] = x1;
                heatmaps_reshape_ex[i + 1][j * heatmapWeight + k + 1] = x0;
            }
        }
    }
    for (size_t i = 0; i < NPOINTS; i++) {
        int startIndex = i * heatmapHeight * heatmapWeight;
        for (size_t j = 0; j < heatmapHeight; j++) {
            int middleIndex = j * heatmapWeight;
            for (size_t k = 0; k < heatmapWeight; k++) {
                heatmaps_reshape[i][j * heatmapWeight + k] =
                    0.5 * (heatmaps_reshape[i][j * heatmapWeight + k] + heatmaps_reshape_ex[i][j * heatmapWeight + k]);
                batch_heatmaps[i][j][k] = 0.5 * (batch_heatmaps[i][j][k] + batch_heatmaps_ex[i][j][k]);
            }
        }
    }
}
void SimplePOSEMindsporePost::GetValidDetBoxes(const std::vector<TensorBase>& tensors,
    const std::vector<TensorBase>& tensors1, std::vector<float> *preds_result,
    uint32_t heatmapHeight, uint32_t heatmapWeight,
    const float center[], const float scale[]) {
    LogInfo << "Begin to GetValidDetBoxes.";
    // auto* bboxPtr = (float*)GetBuffer(tensors[OUTPUT_BBOX_INDEX], batchNum);
    get_data(tensors, tensors1, heatmapHeight, heatmapWeight);
    float maxvals[NPOINTS] = {};
    int idx[NPOINTS] = {};
    for (size_t i = 0; i < NPOINTS; i++) {
        maxvals[i] = get_float_data(i, heatmaps_reshape);
        idx[i] = get_int_data(i, heatmaps_reshape);
    }
    float preds[NPOINTS][2] = {};
    for (size_t i = 0; i < NPOINTS; i++) {
        preds[i][0] = (idx[i]) % heatmapWeight;
        preds[i][1] = floor(idx[i] / heatmapWeight);
        if (maxvals[i] < 0) {
            preds[i][0] = preds[i][0] * (-1);
            preds[i][1] = preds[i][0] * (-1);
        }
    }
    for (size_t i = 0; i < NPOINTS; i++) {
        float hm[64][48] = {};
        for (size_t m = 0; m < 64; m++) {
            for (size_t n = 0; n < 48; n++) {
                hm[m][n] = batch_heatmaps[i][m][n];
            }
        }
        int px = static_cast<int>(floor(preds[i][0] + 0.5));
        int py = static_cast<int>(floor(preds[i][1] + 0.5));
        if (px > 1 && px < heatmapWeight - 1 && py>1 && py < heatmapHeight - 1) {
            float diff_x = hm[py][px + 1] - hm[py][px - 1];
            float diff_y = hm[py + 1][px] - hm[py - 1][px];
            if (diff_x > 0) {
                preds[i][0] = preds[i][0] + 0.25;
            }
            if (diff_x < 0) {
                preds[i][0] = preds[i][0] - 0.25;
            }
            if (diff_y > 0) {
                preds[i][1] = preds[i][1] + 0.25;
            }
            if (diff_y < 0) {
                preds[i][1] = preds[i][1] - 0.25;
            }
        }
    }
    cv::Mat warp_mat(2, 3, CV_32FC1);
    get_affine(center, scale, &warp_mat);
    for (size_t i = 0; i < NPOINTS; i++) {
        preds[i][0] = preds[i][0] * warp_mat.at<double>(0, 0) +
            preds[i][1] * warp_mat.at<double>(0, 1) + warp_mat.at<double>(0, 2);
        preds[i][1] = preds[i][0] * warp_mat.at<double>(1, 0) +
            preds[i][1] * warp_mat.at<double>(1, 1) + warp_mat.at<double>(1, 2);
    }
    for (size_t i = 0; i < NPOINTS; i++) {
        preds_result->push_back(preds[i][0]);
        preds_result->push_back(preds[i][1]);
        preds_result->push_back(maxvals[i]);
    }
}

void SimplePOSEMindsporePost::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
    const std::vector<TensorBase>& tensors1,
    std::vector<std::vector<float> >* node_score_list, const float center[], const float scale[]) {
    LogDebug << "SimplePOSEMindsporePost start to write results.";
    auto shape = tensors[OUTPUT_BBOX_INDEX].GetShape();
    uint32_t batchSize = shape[0];
    uint32_t heatmapHeight = shape[2];
    uint32_t heatmapWeight = shape[3];
    for (uint32_t i = 0; i < batchSize; ++i) {
        std::vector<float> preds_result;
        GetValidDetBoxes(tensors, tensors1, &preds_result, heatmapHeight, heatmapWeight, center, scale);
        node_score_list->push_back(preds_result);
    }

    LogDebug << "SimplePOSEMindsporePost write results successeded.";
}

APP_ERROR SimplePOSEMindsporePost::selfProcess(const float center[], const float scale[],
    const std::vector<TensorBase> &tensors, const std::vector<TensorBase>& tensors1,
    std::vector<std::vector<float> >* node_score_list) {
    LogDebug << "Begin to process SimplePOSEMindsporePost.";
    auto inputs = tensors;
    auto inputs1 = tensors1;
    APP_ERROR ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed, ret=" << ret;
        return ret;
    }
    ret = CheckAndMoveTensors(inputs1);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed, ret=" << ret;
        return ret;
    }
    ObjectDetectionOutput(inputs, inputs1, node_score_list, center, scale);
    LogInfo << "End to process SimplePOSEMindsporePost.";
    return APP_ERR_OK;
}

extern "C" {
std::shared_ptr<MxBase::SimplePOSEMindsporePost> GetObjectInstance() {
    LogInfo << "Begin to get SimplePOSEMindsporePost instance.";
    auto instance = std::make_shared<SimplePOSEMindsporePost>();
    LogInfo << "End to get SimplePOSEMindsporePost Instance";
    return instance;
}
}
}  // namespace MxBase

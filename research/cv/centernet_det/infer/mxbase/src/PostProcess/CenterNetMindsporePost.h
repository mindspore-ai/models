/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef CENTERNET_MINSPORE_PORT_H
#define CENTERNET_MINSPORE_PORT_H
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace MxBase {

class CenterNetMindsporePost : public ObjectPostProcessBase {
 public:
    CenterNetMindsporePost() = default;

    ~CenterNetMindsporePost() = default;

    CenterNetMindsporePost(const CenterNetMindsporePost &other) = default;

    CenterNetMindsporePost &operator=(const CenterNetMindsporePost &other);

    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

    APP_ERROR DeInit() override;

    APP_ERROR Process(const std::vector<TensorBase> &tensors, std::vector<std::vector<ObjectInfo>> &objectInfos,
                      const std::vector<ResizedImageInfo> &resizedImageInfos = {},
                      const std::map<std::string, std::shared_ptr<void>> &configParamMap = {}) override;
    bool IsValidTensors(const std::vector<TensorBase> &tensors) const override;

 private:
    void Resize_Affine(const cv::Mat &srcDet, cv::Mat &dstDet,
                       const ResizedImageInfo &resizedImageInfos);
    void affine_transform(const cv::Mat &A, const cv::Mat &B, cv::Mat &dst);
    void soft_nms(cv::Mat &src, int s, const float sigma, const float Nt, const float threshold);
    void sort_id(float src[][6], const int sum);
    void set_nms(float data[][6], int (*p)[2], const int num);
    void ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                               std::vector<std::vector<ObjectInfo>> &objectInfos,
                               const std::vector<ResizedImageInfo> &resizedImageInfos);

    void GetValidDetBoxes(const std::vector<TensorBase> &tensors, std::vector<DetectBox> &detBoxes,
                          const ResizedImageInfo &resizedImageInfos, uint32_t batchNum);

    void ConvertObjInfoFromDetectBox(std::vector<DetectBox> &detBoxes, std::vector<ObjectInfo> &objectInfos,
                                     const ResizedImageInfo &resizedImageInfo);

    APP_ERROR ReadConfigParams();

 private:
    const uint32_t DEFAULT_CLASS_NUM_MS = 80;
    const uint32_t DEFAULT_RPN_MAX_NUM_MS = 100;

    uint32_t classNum_ = DEFAULT_CLASS_NUM_MS;
    uint32_t rpnMaxNum_ = DEFAULT_RPN_MAX_NUM_MS;
};

extern "C" {
std::shared_ptr<MxBase::CenterNetMindsporePost> GetObjectInstance();
}
}  // namespace MxBase
#endif  // CENTERNET_MINSPORE_PORT_H

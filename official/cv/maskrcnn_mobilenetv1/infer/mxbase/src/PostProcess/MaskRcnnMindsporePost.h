/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. 
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

#ifndef MASKRCNN_MINSPORE_PORT_H
#define MASKRCNN_MINSPORE_PORT_H
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

class MaskRcnnMindsporePost : public ObjectPostProcessBase {
 public:
    MaskRcnnMindsporePost() = default;

    ~MaskRcnnMindsporePost() = default;

    MaskRcnnMindsporePost(const MaskRcnnMindsporePost &other) = default;

    MaskRcnnMindsporePost &operator=(const MaskRcnnMindsporePost &other);

    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

    APP_ERROR DeInit() override;

    APP_ERROR Process(const std::vector<TensorBase> &tensors, std::vector<std::vector<ObjectInfo>> &objectInfos,
                      const std::vector<ResizedImageInfo> &resizedImageInfos = {},
                      const std::map<std::string, std::shared_ptr<void>> &configParamMap = {}) override;

    bool IsValidTensors(const std::vector<TensorBase> &tensors) const;

 private:
    void ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                               std::vector<std::vector<ObjectInfo>> &objectInfos,
                               const std::vector<ResizedImageInfo> &resizedImageInfos);

    void GetValidDetBoxes(const std::vector<TensorBase> &tensors, std::vector<DetectBox> &detBoxes,
                          const uint32_t batchNum);

    APP_ERROR MaskPostProcess(ObjectInfo &objInfo, void *maskPtr, const ResizedImageInfo &imgInfo);

    void ConvertObjInfoFromDetectBox(std::vector<DetectBox> &detBoxes, std::vector<ObjectInfo> &objectInfos,
                                     const ResizedImageInfo &resizedImageInfos);

    APP_ERROR GetMaskSize(const ObjectInfo &objInfo, const ResizedImageInfo &imgInfo, cv::Size &maskSize);

    APP_ERROR ReadConfigParams();

 private:
    const uint32_t DEFAULT_CLASS_NUM_MS_MASK = 80;
    const float DEFAULT_SCORE_THRESH_MS_MASK = 0.7;
    const float DEFAULT_IOU_THRESH_MS_MASK = 0.5;
    const uint32_t DEFAULT_RPN_MAX_NUM_MS_MASK = 1000;
    const uint32_t DEFAULT_MAX_PER_IMG_MS_MASK = 128;
    const float DEFAULT_THR_BINARY_MASK = 0.5;
    const uint32_t DEFAULT_MASK_SIZE_MS_MASK = 28;

    uint32_t classNum_ = DEFAULT_CLASS_NUM_MS_MASK;
    float scoreThresh_ = DEFAULT_SCORE_THRESH_MS_MASK;
    float iouThresh_ = DEFAULT_IOU_THRESH_MS_MASK;
    uint32_t rpnMaxNum_ = DEFAULT_RPN_MAX_NUM_MS_MASK;
    uint32_t maxPerImg_ = DEFAULT_MAX_PER_IMG_MS_MASK;
    float maskThrBinary_ = DEFAULT_THR_BINARY_MASK;
    uint32_t maskSize_ = DEFAULT_MASK_SIZE_MS_MASK;
};

extern "C" {
std::shared_ptr<MxBase::MaskRcnnMindsporePost> GetObjectInstance();
}
}  // namespace MxBase
#endif  // MASKRCNN_MINSPORE_PORT_H

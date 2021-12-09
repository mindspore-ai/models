/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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

#ifndef SSDGHOST_MINSPORE_PORT_H
#define SSDGHOST_MINSPORE_PORT_H

#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace MxBase {
    const int DEFAULT_OBJECT_NUM = 1917;
    const float DEFAULT_IOU_THRESH = 0.6;
    const int DEFAULT_OBJECT_BBOX_TENSOR = 0;
    const int DEFAULT_OBJECT_CONFIDENCE_TENSOR = 1;
    const int DEFAULT_MAX_BBOX_PER_CLASS = 100;

class SsdGhostPostProcess:public ObjectPostProcessBase {
 public:
    SsdGhostPostProcess() = default;

    ~SsdGhostPostProcess() = default;

    SsdGhostPostProcess(const SsdGhostPostProcess &other) = default;

    SsdGhostPostProcess &operator=(const SsdGhostPostProcess &other);

    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

    APP_ERROR DeInit() override;

    APP_ERROR Process(const std::vector<TensorBase> &tensors,
                    std::vector<std::vector<ObjectInfo>> &objectInfos,
                        const std::vector<ResizedImageInfo> &resizedImageInfos = {},
                        const std::map<std::string, std::shared_ptr<void>> &configParamMap = {}) override;

    bool IsValidTensors(const std::vector<TensorBase> &tensors) const override;

 private:
    void ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                                std::vector<std::vector<ObjectInfo>> &objectInfos,
                                const std::vector<ResizedImageInfo> &resizedImageInfos);
    void NonMaxSuppression(std::vector<MxBase::DetectBox>& detBoxes,
        TensorBase &bboxTensor, TensorBase &confidenceTensor, int stride,
        const ResizedImageInfo &imgInfo,
        uint32_t batchNum, uint32_t batchSize);
    void NmsSort(std::vector<DetectBox>& detBoxes, float iouThresh, IOUMethod method);
    void FilterByIou(std::vector<DetectBox> dets,
            std::vector<DetectBox>& sortBoxes, float iouThresh, IOUMethod method);
    float CalcIou(DetectBox boxA, DetectBox boxB, IOUMethod method);

    int objectNum_ = DEFAULT_OBJECT_NUM;
    float iouThresh_ = DEFAULT_IOU_THRESH;
    int objectBboxTensor_ = DEFAULT_OBJECT_BBOX_TENSOR;
    int objectConfidenceTensor_ = DEFAULT_OBJECT_CONFIDENCE_TENSOR;
    int maxBboxPerClass_ = DEFAULT_MAX_BBOX_PER_CLASS;
};

    extern "C" {
        std::shared_ptr<MxBase::SsdGhostPostProcess> GetObjectInstance();
    }
}  // namespace MxBase
#endif

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

#ifndef SimplePOSE_MINSPORE_PORT_H
#define SimplePOSE_MINSPORE_PORT_H
#include <algorithm>
#include <vector>
#include <map>
#include<string>
#include<memory>
#include <opencv4/opencv2/opencv.hpp>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace MxBase {

class SimplePOSEMindsporePost : public ObjectPostProcessBase {
 public:
    SimplePOSEMindsporePost() = default;

    ~SimplePOSEMindsporePost() = default;

    SimplePOSEMindsporePost(const SimplePOSEMindsporePost &other) = default;

    SimplePOSEMindsporePost &operator=(const SimplePOSEMindsporePost &other);

    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

    APP_ERROR DeInit() override;

    APP_ERROR selfProcess(const float center[], const float scale[], const std::vector<TensorBase> &tensors,
        const std::vector<TensorBase>& tensors1,
        std::vector<std::vector<float> >* node_score_list);

    bool IsValidTensors(const std::vector<TensorBase> &tensors) const override;

 private:
    void ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
        const std::vector<TensorBase>& tensors1,
        std::vector<std::vector<float> > *node_score_list, const float center[], const float scale[]);

    void GetValidDetBoxes(const std::vector<TensorBase>& tensors,
        const std::vector<TensorBase>& tensors1, std::vector<float> *preds_result,
        uint32_t heatmapHeight, uint32_t heatmapWeight,
        const float center[], const float scale[]);
};

extern "C" {
std::shared_ptr<MxBase::SimplePOSEMindsporePost> GetObjectInstance();
}
}  // namespace MxBase
#endif  // SimplePOSE_MINSPORE_PORT_H

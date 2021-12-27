/*
 * Copyright(C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxPlugins/ModelPostProcessors/ModelPostProcessorBase/MxpiModelPostProcessorBase.h"
#include "infer/mxbase/SSDPostProcessor.h"

class MxpiSSDMobileNetV2PostProcess
    : public MxPlugins::MxpiModelPostProcessorBase {
    APP_ERROR Init(const std::string &configPath, const std::string &labelPath,
                   MxBase::ModelDesc modelDesc) override;
    APP_ERROR DeInit() override;
    APP_ERROR
    Process(std::shared_ptr<void> &metaDataPtr,
            MxBase::PostProcessorImageInfo postProcessorImageInfo,
            std::vector<MxTools::MxpiMetaHeader> &headerVec,
            std::vector<std::vector<MxBase::BaseTensor>> &tensors) override;
    std::vector<MxBase::ObjectInfo> nms(
        const std::vector<MxBase::ObjectInfo> &object_infos, float thres,
        int max_boxes);
    float iou(const MxBase::ObjectInfo &a, const MxBase::ObjectInfo &b);

 private:
    APP_ERROR CheckModelCompatibility();
};

extern "C" {
std::shared_ptr<MxpiSSDMobileNetV2PostProcess> GetInstance();

std::shared_ptr<sdk_infer::mxbase_infer::SSDPostProcessor> GetObjectInstance();
}

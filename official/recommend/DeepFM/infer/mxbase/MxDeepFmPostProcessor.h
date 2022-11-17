/*
 * Copyright(C) 2021 Huawei Technologies Co., Ltd. All rights reserved.
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

#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"

class MSDeepfmPostProcessor : public MxBase::ObjectPostProcessorBase {
 public:
    APP_ERROR Init(const std::string &configPath, const std::string &labelPath,
                   MxBase::ModelDesc modelDesc) override {
        APP_ERROR ret = APP_ERR_OK;

        this->GetModelTensorsShape(modelDesc);

        ret = check_ms_model_compatibility();

        if (ret != APP_ERR_OK) {
            LogError << "Failed when checking model compatibility";
            return ret;
        }

        // init for this class derived information
        ret = read_config_params();
        return ret;
    }

    /*
     * @description: Get the info of detected object from output and resize to
     * original coordinates.
     * @param featLayerData Vector of output feature data.
     * @param objInfos Address of output object infos.
     * @param useMpPictureCrop if true, offsets of coordinates will be given.
     * @param postImageInfo Info of model/image width and height, offsets of
     * coordinates.
     * @return: ErrorCode.
     */
    APP_ERROR Process(std::vector<std::shared_ptr<void>> &featLayerData,
                      std::vector<ObjDetectInfo> &objInfos,
                      const bool useMpPictureCrop,
                      MxBase::PostImageInfo postImageInfo) override {
        if (featLayerData.empty()) {
            LogError << "featLayerData is empty";
            return APP_ERR_INVALID_PARAM;
        }
        auto *predict = static_cast<float *>(featLayerData[0].get());

        std::cout << predict[0] << std::endl;
        return APP_ERR_OK;
    }

 private:
    // check om model output shape compatibility
    APP_ERROR check_ms_model_compatibility() { return APP_ERR_OK; }

    // retrieve this specific config parameters
    APP_ERROR read_config_params() { return APP_ERR_OK; }

 private:
    int m_nTopK_ = 100;
    int m_nHMWidth_ = 208;
    int m_nHMHeight_ = 208;
    int m_nKeyCounts_ = 5;

    float m_fScoreThresh_ = 0.2;
};

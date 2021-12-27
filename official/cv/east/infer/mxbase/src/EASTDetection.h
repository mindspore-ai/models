/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef EAST_RECOGNITION_H
#define EAST_RECOGNITION_H

#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/TextObjectPostProcessBase.h"
#include "PostProcess/EASTMindSporePost.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"


extern std::vector<double> g_inferCost;

struct InitParam {
    uint32_t deviceId;
    std::string nmsThresh;
    std::string scoreThresh;
    bool checkTensor;
    std::string modelPath;

    uint32_t outSize;
};

class EASTDetection {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat *imageMat);
    APP_ERROR Resize(cv::Mat *srcImageMat, cv::Mat *dstImageMat);
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase>& tensors,
                          std::vector<std::vector<MxBase::TextObjectInfo>> *textInfos);
    APP_ERROR WriteResult(const std::vector<std::vector<MxBase::TextObjectInfo>> &textInfos,
                          const std::string &imgPath);
    APP_ERROR Process(const std::string &imgPath);
    // get infer time
    double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}

 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::EASTPostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    uint32_t imageWidth_ = 0;
    uint32_t imageHeight_ = 0;
    // infer time
    double inferCostTimeMilliSec = 0.0;
};
#endif

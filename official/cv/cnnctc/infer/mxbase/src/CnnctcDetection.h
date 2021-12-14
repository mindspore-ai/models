/*
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

#ifndef MXBASE_CNNCTC_DETECTION_H
#define MXBASE_CNNCTC_DETECTION_H

#include <vector>
#include <memory>
#include <map>
#include <string>
#include "opencv2/opencv.hpp"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "TextGenerationPostProcessors/CrnnPostProcess.h"
#include "MxBase/CV/Core/DataType.h"

struct InitParam {
    uint32_t deviceId;
    uint32_t classNum;
    uint32_t objectNum;
    uint32_t blankIndex;
    std::string labelPath;
    bool softmax;
    bool checkTensor;
    bool argmax;
    std::string modelPath;
};
class CnnctcDetection {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat *imageMat);
    APP_ERROR ResizeAndPad(const cv::Mat &srcImageMat, MxBase::TensorBase *tensorBase);
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR PostProcess(std::vector<MxBase::TensorBase> *outputs, std::vector<MxBase::TextsInfo> *textInfos);
    APP_ERROR Process(const std::string &imgPath, std::string *result);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::CrnnPostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    uint32_t imageWidth_ = 0;
    uint32_t imageHeight_ = 0;
};

#endif

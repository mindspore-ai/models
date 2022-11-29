/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef CRNN_RECOGNITION_H
#define CRNN_RECOGNITION_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "TextGenerationPostProcessors/CrnnPostProcess.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

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

class CrnnRecognition {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadAndResize(const std::string &imgPath, MxBase::TensorBase &outputTensor);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase>& tensors, std::vector<MxBase::TextsInfo>& textInfos);
    APP_ERROR Process(const std::string &imgPath, std::string &result);
    // get infer time
    double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}
 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::CrnnPostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    // infer time
    double inferCostTimeMilliSec = 0.0;
};
#endif

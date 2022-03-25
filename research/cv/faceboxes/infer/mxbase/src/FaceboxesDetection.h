/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd.
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
#ifndef MXBASE_FACEBOXESDETECTION_H
#define MXBASE_FACEBOXESDETECTION_H

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"


struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

struct ImageShape {
    uint32_t width;
    uint32_t height;
};

class FaceboxesDetection {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR Process(std::string &imgPath);
    APP_ERROR PostProcess(std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR ImagePreprocess(std::string &imgPath);
    APP_ERROR ReadImage(std::string &imgPath, MxBase::DvppDataInfo &output, ImageShape &imgShape);
    APP_ERROR Resize(const MxBase::DvppDataInfo &input, MxBase::TensorBase &outputTensor);

 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
};
#endif

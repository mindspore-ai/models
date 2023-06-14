/*
* Copyright(C) 2022. Huawei Technologies Co.,Ltd
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

#ifndef MXBASE_ERFNETINFER_H
#define MXBASE_ERFNETINFER_H
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

class c3dInfer {
 public:
    c3dInfer(const uint32_t &deviceId, const std::string &modelPath);
    ~c3dInfer();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR Process(const std::string &imgPath);
    APP_ERROR ReadTensorFromFile(const std::string &file, float *data, uint32_t size);

 protected:
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, const MxBase::TensorBase &tensorBase);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
};
#endif

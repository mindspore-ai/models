/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <string>
#include <vector>
#include <memory>
#ifndef MxBase_DEEPSORT_H
#define MxBase_DEEPSORT_H

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string modelPath;
    std::vector<float> MEAN;
    std::vector<float> STD;
};

class DEEPSORT {
 public:
     APP_ERROR Init(const InitParam &initParam);
     APP_ERROR DeInit();
     APP_ERROR ReadInputTensor(const std::string &fileName, MxBase::TensorBase &tensorBase);
     APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
     APP_ERROR PostProcess(std::vector<float> &det, const std::vector<float> &feature);
     APP_ERROR Process(const std::string &imagePath, std::vector<float> &det, const std::string &result_path);
     APP_ERROR SaveInferResult(std::vector<float> &batchFeaturePaths, const std::vector<MxBase::TensorBase> &inputs);
     APP_ERROR WriteResult(const std::vector<float> &outputs, const std::string &result_path);
     double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}
 private:
     std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
     MxBase::ModelDesc modelDesc_;
     uint32_t deviceId_ = 0;
     double inferCostTimeMilliSec = 0.0;
};
#endif

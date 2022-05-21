/*
 * Copyright 2022 Huawei Technologies Co., Ltd.
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
#ifndef MxBase_TGCN_H
#define MxBase_TGCN_H

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string modelPath;
    std::string dataset;
};

class TGCN {
 public:
     APP_ERROR Init(const InitParam &initParam);
     APP_ERROR DeInit();
     APP_ERROR VectorToTensorBase(const std::string &dataset, const std::vector<std::vector<float>> &input_x,
                                  MxBase::TensorBase *tensorBase);
     APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
     APP_ERROR Process(const std::string &dataset, const std::vector<std::vector<float>> &input_x,
                       const InitParam &initParam, std::vector<float> *output);
     APP_ERROR SaveInferResult(std::vector<float> *batchFeaturePaths,
                               const std::vector<MxBase::TensorBase> &inputs);
 private:
    //  std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
     std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
     MxBase::ModelDesc modelDesc_;
     uint32_t deviceId_ = 0;
     double inferCostTimeMilliSec = 0.0;
};
#endif

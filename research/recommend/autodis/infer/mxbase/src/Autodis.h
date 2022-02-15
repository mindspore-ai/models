/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef MxBase_AUTODIS_H
#define MxBase_AUTODIS_H

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string modelPath;
};

class AUTODIS {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    template<class dtype>
    APP_ERROR VectorToTensorBase(const std::vector<std::vector<dtype>> &input_x, uint32_t inputId
                                    , MxBase::TensorBase &tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR Process(const std::vector<std::vector<int>> &ids, const std::vector<std::vector<float>> &wts
                        , const std::vector<std::vector<float>> &label, const InitParam &initParam,
                        std::vector<int> &pred, std::vector<float> &probs);
    APP_ERROR PostProcess(std::vector<float> &probs, const std::vector<MxBase::TensorBase> &inputs);
    APP_ERROR PrintInputInfo(std::vector<MxBase::TensorBase> inputs);
    double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}
 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    double inferCostTimeMilliSec = 0.0;
};
#endif

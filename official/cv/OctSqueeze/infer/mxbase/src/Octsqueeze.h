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

#ifndef MXBASE_OCTSQUEEZE_H
#define MXBASE_OCTSQUEEZE_H
#include <memory>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "acl/acl.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/CV/Core/DataType.h"

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

class octsqueeze {
 public:
  APP_ERROR Init(const InitParam &initParam);
  APP_ERROR DeInit();
  APP_ERROR VectorToTensorBase_float32(std::vector<std::vector<float>> &batchFeatureVector,
                                    MxBase::TensorBase &tensorBase);
  APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> &outputs);
  APP_ERROR Process(std::vector<std::vector<float>> &input, const InitParam &initParam,
                          float* &outputs);
  // get infer time
  double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}


 private:
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_octsqueeze;
  MxBase::ModelDesc modelDesc_;
  uint32_t deviceId_ = 0;
  // infer time
  double inferCostTimeMilliSec = 0.0;
};

#endif



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

#ifndef MXBASE_Tbnet_H
#define MXBASE_Tbnet_H
#include <memory>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/CV/Core/DataType.h"

struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string modelPath;
};

class Tbnet {
 public:
  APP_ERROR Init(const InitParam &initParam);
  APP_ERROR DeInit();
  APP_ERROR VectorToTensorBase_int(const std::vector<std::vector<int64_t>> &input, MxBase::TensorBase &tensorBase,
                               const std::vector<uint32_t> &shape);
  APP_ERROR VectorToTensorBase_float(const std::vector<std::vector<float>> &input, MxBase::TensorBase &tensorBase,
                               const std::vector<uint32_t> &shape);
  APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
  APP_ERROR Process(const int &index, const std::string &datapath,
                    const InitParam &initParam, std::vector<int> &outputss);
  APP_ERROR ReadBin_int(const std::string &path, std::vector<std::vector<int64_t>> &dataset,
                    const int shape);
  APP_ERROR ReadBin_float(const std::string &path, std::vector<std::vector<float>> &dataset,
                    const int shape);
  // get infer time
  double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}


 private:
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_Tbnet;
  MxBase::ModelDesc modelDesc_;
  uint32_t deviceId_ = 0;
  // infer time
  double inferCostTimeMilliSec = 0.0;
};

#endif

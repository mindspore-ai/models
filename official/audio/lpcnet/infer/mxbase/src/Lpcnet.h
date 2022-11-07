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

#ifndef MXBASE_Lpcnet_H
#define MXBASE_Lpcnet_H
#include <memory>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string encoder_modelPath;
    std::string decoder_modelPath;
    std::string inferSrcTokensPath;
    std::string resultName;
    int begin;
    int end;
};

class Lpcnet {
 public:
  APP_ERROR Init(const InitParam &initParam);
  APP_ERROR DeInit();
  void VectorToTensorBase_Int32(const std::vector<std::vector<int>> &batchFeatureVector,
                                MxBase::TensorBase &tensorBase);
  void VectorToTensorBase_Float32(const std::vector<std::vector<float>> &batchFeatureVector,
                                       MxBase::TensorBase &tensorBase);
  APP_ERROR Inference_Encoder(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
  APP_ERROR Inference_Decoder(std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
  APP_ERROR Process(const std::vector<std::vector<float>> &cfeat, const std::vector<std::vector<int>> &period,
                    const std::vector<std::vector<float>> &feature, const InitParam &initParam,
                    std::vector<int16_t> &mem_outputs);
  std::vector<std::vector<float>> make_zero_martix_float(int m, int n);
  std::vector<std::vector<double>> make_zero_martix_double(int m, int n);
  std::vector<std::vector<int>> make_128_martix_int(int m, int n);
  void process_decoder(double *new_p, const std::vector<std::vector<float>> &feature, int fr);
  // get infer time
  double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}


 private:
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_encoder;
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_decoder;

  MxBase::ModelDesc modelDesc_encoder;
  MxBase::ModelDesc modelDesc_decoder;
  uint32_t deviceId_ = 0;
  // infer time
  double inferCostTimeMilliSec = 0.0;
};

#endif

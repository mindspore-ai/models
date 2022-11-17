/**
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

#ifndef SENTIMENTNET_H
#define SENTIMENTNET_H

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

extern std::vector<double> g_infer_cost;
extern uint32_t g_true_positive;
extern uint32_t g_false_positive;
extern uint32_t g_true_negative;
extern uint32_t g_false_negative;

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

class SentimentNet {
 public:
    APP_ERROR Init(const InitParam &initParam);

    APP_ERROR DeInit();

    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
                        const std::shared_ptr<std::vector<MxBase::TensorBase>> &outputs);

    APP_ERROR Process(const std::string &inferPath, const std::string &fileName, bool firstInput,
                      bool eval, const std::vector<uint32_t> &labels, const uint32_t startIndex);

    APP_ERROR PostProcess(const std::shared_ptr<std::vector<MxBase::TensorBase>> &outputs,
                          const std::shared_ptr<std::vector<uint32_t>> &argmax, bool printResult);

 protected:
    APP_ERROR ReadTensorFromFile(const std::string &file, uint32_t *data, const uint32_t size);

    APP_ERROR ReadInputTensor(const std::string &fileName, uint32_t index,
                              const std::shared_ptr<std::vector<MxBase::TensorBase>> &inputs);

    APP_ERROR WriteResult(const std::string &fileName, const std::vector<uint32_t> &argmax, bool firstInput);

    void
    CountPredictResult(const std::vector<uint32_t> &labels, uint32_t startIndex, const std::vector<uint32_t> &argmax);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
};

#endif

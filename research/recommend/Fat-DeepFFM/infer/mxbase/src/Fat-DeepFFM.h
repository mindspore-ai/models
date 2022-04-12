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

#ifndef MXBASE_BERTBASE_H
#define MXBASE_BERTBASE_H

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

extern std::vector<double> g_inferCost;
extern uint32_t g_TP;
extern uint32_t g_FP;
extern uint32_t g_FN;

struct InitParam {
    uint32_t deviceId;
    std::string datasetPath;
    std::string modelPath;
};


class DeepFFMBase {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR Process(const std::string &inferPath, const std::string &fileName);
    APP_ERROR PostProcess(std::vector<MxBase::TensorBase> *outputs, std::vector<float> *result);
 protected:
    APP_ERROR ReadTensorFromFile(const std::string &file, float *data, uint32_t size);
    APP_ERROR ReadInputTensor(const std::string &fileName, uint32_t index, std::vector<MxBase::TensorBase> *inputs,
                              uint32_t size, MxBase::TensorDataType type);
    APP_ERROR getAucResult(const std::string &labelFile, std::vector<float> *result);
 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    std::vector<std::string> labelMap_ = {};
    uint32_t deviceId_ = 0;
};
#endif

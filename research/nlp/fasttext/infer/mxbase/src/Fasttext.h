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

#ifndef MXBASE_FASTTEXTBASE_H
#define MXBASE_FASTTEXTBASE_H

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

extern std::vector<double> g_inferCost;

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
    std::string inferSrcTokensPath;
    std::string resultName;
};

enum DataIndex {
    INPUT_SRCTOKENS = 0,
    INPUT_SRCTOKENSLENGTH = 1,
};

class FasttextNerBase {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR Process(const std::string &inferSrcTokensPath, const std::string &fileName);
    APP_ERROR PostProcess(std::vector<MxBase::TensorBase> *outputs, std::vector<uint32_t> *predict);

 protected:
    APP_ERROR ReadInputTensor(int32_t *data, uint32_t index, std::vector<MxBase::TensorBase> *inputs,
                              const uint32_t size);
    APP_ERROR WriteResult(const std::string &fileName, const std::vector<uint32_t> &predict);

 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    uint32_t maxLength_ = 467;
    std::string resultName_ = "";
};
#endif

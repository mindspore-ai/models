/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd.
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
#ifndef SPHEREFACE_MINSPORE_PORT_H
#define SPHEREFACE_MINSPORE_PORT_H
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "MxBase/CV/Core/DataType.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    bool softmax;
    bool checkTensor;
    uint32_t deviceId;
    std::string CoarseModelPath;
    std::string FineModelPath;
};

class DepthNetOpencv {
 public:
    APP_ERROR Init(const InitParam& initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase>& inputs,
        std::vector<MxBase::TensorBase>* outputs, const int flag);
    APP_ERROR CoarseProcess(const std::string& inferPath, const std::string& RgbBinFile);
    APP_ERROR FineProcess(const std::string& inferPath, const std::string& RgbBinFile,
        const std::string& CoarseDepthBinFile);

    // additional API
    APP_ERROR ReadTensorFromFile(const std::string& file, float* data, const int flag);
    APP_ERROR ReadInputTensor(const std::string& fileName,
        std::vector<MxBase::TensorBase>* inputs, const int flag);
    APP_ERROR SaveResult(const std::string& result, std::vector<MxBase::TensorBase> outputs);

    // get infer time
    double GetInferCostMilliSec() const { return inferCostTimeMilliSec; }

 private:
    uint32_t deviceId_ = 0;
    std::shared_ptr<MxBase::ModelInferenceProcessor> coarseModel_;
    MxBase::ModelDesc coarseModelDesc_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> fineModel_;
    MxBase::ModelDesc fineModelDesc_;
    std::vector<uint32_t> rgbInputDataShape_ = { 1, 3, 228, 304 };
    uint32_t rgbInputDataSize_ = 207936;
    std::vector<uint32_t> depthInputDataShape_ = { 1, 1, 55, 74 };
    uint32_t depthInputDataSize_ = 4070;
    // infer time
    double inferCostTimeMilliSec = 0.0;
};

#endif

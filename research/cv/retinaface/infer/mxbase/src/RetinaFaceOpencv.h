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
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    bool softmax;
    bool checkTensor;
    uint32_t deviceId;
    std::string modelPath;
    std::string resultPath;
};

class RetinaFaceOpencv {
 public:
    APP_ERROR Init(const InitParam& initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase>& inputs,
        std::vector<MxBase::TensorBase>* outputs);
    APP_ERROR Process(const std::string& inferPath, const std::string& fileName);

    // additional API
    APP_ERROR ReadTensorFromFile(const std::string& file, float* data);
    APP_ERROR ReadInputTensor(const std::string& fileName,
        std::vector<MxBase::TensorBase>* inputs);

    // get infer time
    double GetInferCostMilliSec() const { return inferCostTimeMilliSec; }


 private:
    APP_ERROR SaveResult(const std::string& result,
    std::vector<MxBase::TensorBase> outputs);

 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    std::vector<uint32_t> inputDataShape_ = {1, 3, 5568, 1056};
    uint32_t inputDataSize_ = 17639424;
    std::string resultPath_ = "./mxbase_out/";
    // infer time
    double inferCostTimeMilliSec = 0.0;
};

#endif

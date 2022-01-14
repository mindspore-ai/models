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

#ifndef PGAN_H
#define PGAN_H

#include <dirent.h>
#include <math.h>
#include <memory>
#include <vector>
#include <map>
#include <random>
#include <ctime>
#include <string>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "MxBase/Log/Log.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/DeviceManager/DeviceManager.h"

struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string modelPath;
    std::string srPath;
};


class Pgan {
 public:
     APP_ERROR Init(const InitParam &initParam, uint32_t batchSize);
     APP_ERROR DeInit();
     APP_ERROR BuildNoiseData(uint32_t batchSize, uint32_t inputRound, std::vector<MxBase::TensorBase> &inputs);
     APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs,
                         uint32_t inputBatch);
     APP_ERROR PostProcess(std::vector<MxBase::TensorBase> &outputs, cv::Mat &resultImg, uint32_t size1,
                           uint32_t size2);
     APP_ERROR Process(uint32_t batchSize);
     // get infer time
     double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}
 private:
     APP_ERROR SaveResult(cv::Mat &resultImg, std::string imgName);
     std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
     std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
     std::string srPath_;
     MxBase::ModelDesc modelDesc_;
     uint32_t deviceId_ = 0;
     // infer time
     double inferCostTimeMilliSec = 0.0;
};


#endif

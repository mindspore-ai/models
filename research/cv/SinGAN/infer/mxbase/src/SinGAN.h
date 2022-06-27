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
#ifndef SINGAN_H
#define SINGAN_H
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include <opencv2/opencv.hpp>

extern std::vector<double> g_infer_cost;
extern uint32_t g_total;
extern uint32_t g_total_acc;

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    std::string modelPath;
    uint32_t classNum;
    std::string outputDataPath;
};

class SinGAN {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR Process(const std::string &outPath, const std::string &inferPath, const std::string &fileName1,
                     const std::string &fileName2);
    APP_ERROR WriteResult(std::vector<MxBase::TensorBase> outputs, cv::Mat *resultImg);
 protected:
    APP_ERROR ReadTensorFromFile(const std::string &file,  float *data);
    APP_ERROR ReadInputTensor(const std::string &fileName, uint32_t index, std::vector<MxBase::TensorBase> *inputs,
                           const std::string &dataName);
 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    std::vector<std::string> labelMap_ = {};
    uint32_t deviceId_ = 0;
    uint32_t classNum_ = 0;
    std::string outputDataPath_ = "./result";
    std::vector<uint32_t> inputDataShape_ = {1, 3, 169, 250};
    uint32_t inputDataSize_ = 750000;
};
#endif


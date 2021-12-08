/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MxBase_STGCN_H
#define MxBase_STGCN_H
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string modelPath;
    int input_size;
    int scale_size;
    int test_crops;
    int length;
    std::vector<int> input_mean;
    std::vector<int> input_std;
    std::string modality;
    std::string data_path;
};

class TSN {
 public:
     APP_ERROR Init(const InitParam &initParam);
     APP_ERROR DeInit();
     APP_ERROR GroupScale(std::vector<cv::Mat> *images, const InitParam &initParam);
     APP_ERROR GroupCenterCrop(std::vector<cv::Mat> *images, const InitParam &initParam);
     APP_ERROR SaveInferResult(std::vector<std::vector<float>> *batchFeaturePaths,
      const std::vector<MxBase::TensorBase> &inputs);
     APP_ERROR CVMatToTensorBase(float* data, MxBase::TensorBase *tensorBase, const InitParam &initParam);
     APP_ERROR GroupNormalize(std::vector<cv::Mat> *images, const InitParam &initParam);
     APP_ERROR GroupOverSample(std::vector<cv::Mat> *images, const InitParam &initParam);
     APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
     APP_ERROR PostProcess(const std::vector<std::vector<float>> &result);
     APP_ERROR Process(const std::string &dataPath, const std::string &image_tmpl,
      const std::vector<int> &indices, const InitParam &initParam);
     double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}
 private:
     std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
     MxBase::ModelDesc modelDesc_;
     uint32_t deviceId_ = 0;
     double inferCostTimeMilliSec = 0.0;
};
#endif

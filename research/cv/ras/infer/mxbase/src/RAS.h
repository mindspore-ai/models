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

#ifndef RAS_H
#define RAS_H

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

extern std::vector<double> g_inferCost;

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
    std::string outputDataPath;
};

class RAS {
 public:
    static const int IMG_H = 352;
    static const int IMG_W = 352;
    static const int IMG_C = 3;
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR Process(const std::string &inferPath, std::string &fileName);

 protected:
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &img);
    APP_ERROR ImageLoader(const std::string &imgPath, cv::Mat &img);
    APP_ERROR PreProcess(const std::string &imgPath, cv::Mat &img, float (&img_array)[IMG_C][IMG_H][IMG_W]);
    APP_ERROR ArrayToTensorBase(float (&imageArray)[IMG_C][IMG_H][IMG_W], MxBase::TensorBase *tensorBase);
    APP_ERROR ReadResult(void);
    APP_ERROR PostProccess(std::vector<MxBase::TensorBase> &outputs, const std::string &imgPath,
                           const std::string &fileName);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    std::string outputDataPath_ = "../results/results.txt";
    std::vector<uint32_t> inputDataShape_ = {1, 3, 352, 352};
    uint32_t inputDataSize_ = 371712;
};

#endif

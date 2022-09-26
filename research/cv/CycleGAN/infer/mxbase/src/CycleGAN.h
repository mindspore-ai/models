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
 * ============================================================================
 */

#ifndef MXBASE_DCGAN_H
#define MXBASE_DCGAN_H

#include <dirent.h>

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

extern std::vector<double> g_inferCost;
const uint32_t FLOAT32_TYPE_BYTE_NUM = 4;
const float NORMALIZE_MEAN = 255 / 2.0;
const float NORMALIZE_STD = 255 / 2.0;
const uint32_t CHANNEL = 3;

struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string modelPath;
    std::string savePath;
    std::string dataPath;
    uint32_t imageWidth;
    uint32_t imageHeight;
};
InitParam initParam_;

class CycleGAN {
 public:
    CycleGAN();
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(std::vector<MxBase::TensorBase> outputs, cv::Mat *resultMat);
    APP_ERROR Process(const std::string &imgPath, const std::string &imgName);
    APP_ERROR SaveResult(const cv::Mat &resultMat, const std::string &imgName);
    APP_ERROR Resize(cv::Mat &srcImageMat, cv::Mat &dstImageMat);
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t originalWidth_;
    uint32_t originalHeight_;
};
#endif

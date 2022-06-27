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

#ifndef MXBASE_NIMA_H
#define MXBASE_NIMA_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

#define CHANNEL 3
#define NET_WIDTH 224
#define NET_HEIGHT 224

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    std::string modelPath;
};

class Nima {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat);
    APP_ERROR ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
    APP_ERROR hwc_to_chw(const cv::Mat &dstImageMat, float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH]);
    APP_ERROR NormalizeImage(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH], \
                             float (&imageArrayNormal)[CHANNEL][NET_HEIGHT][NET_WIDTH]);
    APP_ERROR ArrayToTensorBase(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH], \
                                MxBase::TensorBase *tensorBase);
    APP_ERROR Inference(std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR WriteResult(const std::string &imageFile, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR Process(const std::string &imgPath);
    // get infer time
    double GetInferCostMilliSec() const { return inferCostTimeMilliSec; }

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    // infer time
    double inferCostTimeMilliSec = 0.0;
};

#endif

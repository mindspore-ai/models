/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MXBASE_FACEDETECTION_H
#define MXBASE_FACEDETECTION_H

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

#define CHANNEL 3
#define NET_WIDTH 768
#define NET_HEIGHT 448
#define COLOR_RANGE 255

extern std::vector<double> g_inferCost;

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
    uint32_t classNum;
};

class FaceDetection {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase>* outputs);
    APP_ERROR Process(const std::string &imgName, const std::string &imgDr, const std::string &mxbaseResultPath);
    APP_ERROR WriteResult2Bin(size_t index, MxBase::TensorBase output, std::string resultPathName);

 protected:
    APP_ERROR ReadImage(const std::string &imageName, cv::Mat* imageMat, const std::string &imgDr);
    APP_ERROR ResizeAndPadding(cv::Mat* imageMat);
    APP_ERROR hwc_to_chw(const cv::Mat &imageMat, float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH]);
    APP_ERROR NormalizeImage(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH], \
                             float (&imageArrayNormal)[CHANNEL][NET_HEIGHT][NET_WIDTH]);
    APP_ERROR ArrayToTensorBase(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH], \
                                MxBase::TensorBase* tensorBase);
    APP_ERROR WriteResult(std::vector<MxBase::TensorBase> outputs, \
                          const std::string &imgName, \
                          const std::string &mxbaseResultPath);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
    uint32_t classNum_ = 1;
};
#endif

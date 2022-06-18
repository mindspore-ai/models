/*
 * Copyright 2022. Huawei Technologies Co., Ltd.
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

#ifndef MXBASE_RESNEST50CLASSIFY_H
#define MXBASE_RESNEST50CLASSIFY_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/postprocess/include/ClassPostProcessors/Resnet50PostProcess.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

#define CHANNEL 3
#define NET_HEIGHT 256
#define NET_WIDTH 256
extern std::vector<double> g_inferCost;

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    uint32_t topk;
    bool softmax;
    bool checkTensor;
    std::string modelPath;
};

class ResNeSt50Classify {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat);
    APP_ERROR Resize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
    APP_ERROR Crop(const cv::Mat &srcMat, cv::Mat &dstMat);
    APP_ERROR Transe(const cv::Mat &srcMat, float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH]);
    APP_ERROR ArrayToTensorBase(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH], \
                                MxBase::TensorBase* tensorBase);
    APP_ERROR NormalizeImage(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH], \
                            float (&imageArrayNormal)[CHANNEL][NET_HEIGHT][NET_WIDTH]);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
        std::vector<std::vector<MxBase::ClassInfo>> &clsInfos);
    APP_ERROR Process(const std::string &imgPath);
    APP_ERROR SaveResult(const std::string &imgPath, const std::vector<std::vector<MxBase::ClassInfo>> &BatchClsInfos);
 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::Resnet50PostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    std::ofstream tfile_;
};
#endif


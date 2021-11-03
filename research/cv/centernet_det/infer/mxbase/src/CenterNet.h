/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef CENTERNETPOST_CENTERNET_H
#define CENTERNETPOST_CENTERNET_H
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "PostProcess/CenterNetMindsporePost.h"
#include "MxBase/DeviceManager/DeviceManager.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    uint32_t topk;
    bool softmax;
    bool checkTensor;
    std::string modelPath;
};

struct ImageShape {
    uint32_t width;
    uint32_t height;
};

class CenterNet {
 public:
        APP_ERROR Init(const InitParam &initParam);
        APP_ERROR DeInit();
        APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat, ImageShape &imgShape);
        APP_ERROR Resize_Affine(const cv::Mat &srcImage, cv::Mat &dstImage, ImageShape &imgShape);
        APP_ERROR CVMatToTensorBase(std::vector<float> &imageData, MxBase::TensorBase &tensorBase);
        APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
        APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                          std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                          const std::map<std::string, std::shared_ptr<void>> &configParamMap);
        APP_ERROR Process(const std::string &imgPath, const std::string &resultPath);
        // get infer time
        double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}

 private:
        std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
        std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
        std::shared_ptr<MxBase::CenterNetMindsporePost> post_;
        MxBase::ModelDesc modelDesc_;
        uint32_t deviceId_ = 0;
        // infer time
        double inferCostTimeMilliSec = 0.0;
};

#endif

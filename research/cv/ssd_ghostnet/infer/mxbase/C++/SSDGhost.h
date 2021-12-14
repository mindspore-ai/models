/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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



#ifndef SSD_GHOST
#define SSD_GHOST

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "PostProcess/SsdGhost_MindsporePost.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    float iou_thresh;
    float score_thresh;

    bool checkTensor;
    std::string modelPath;
};

class SSDGhost {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    void ReadImage(const std::string &imgPath, cv::Mat &imageMat);
    void ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
        std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
        const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
        const std::map<std::string, std::shared_ptr<void>> &configParamMap);
    APP_ERROR SaveResult(const std::string &imgPath,
                        std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos);
    APP_ERROR Process(const std::string &imgPath);
 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::SsdGhostPostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
};
#endif

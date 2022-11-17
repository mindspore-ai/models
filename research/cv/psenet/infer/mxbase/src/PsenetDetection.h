/*
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

#ifndef MXBASE_PSENET_DETECTION_H
#define MXBASE_PSENET_DETECTION_H

#include <vector>
#include <memory>
#include <map>
#include <string>
#include "opencv2/opencv.hpp"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "TextObjectPostProcessors/PSENetPostProcess.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/CV/Core/DataType.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;

    uint32_t kernelNum;
    float pseScale;
    float minKernelArea;
    float minScore;
    float minArea;
    bool checkTensor;
    std::string modelPath;
};
class PsenetDetection {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat);
    APP_ERROR Pad(cv::Mat &srcImageMat, cv::Mat &dstImageMat);
    APP_ERROR Resize(cv::Mat &srcImageMat, cv::Mat &dstImageMat);

    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);

    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(std::vector<MxBase::TensorBase> &outputs,
                          std::vector<std::vector<MxBase::TextObjectInfo>> &objInfos);
    APP_ERROR Process(const std::string &imgPath);
    APP_ERROR SaveResult(const std::string &imgPath,
     const std::vector<std::vector<MxBase::TextObjectInfo>> &batchTextObjectInfos);
    void TensorBaseToCVMat(std::vector<cv::Mat> &imageMat,
                                             const MxBase::TensorBase &tensor, int type);
    void growing_text_line(const std::vector<cv::Mat> &kernels,
                                             std::vector<std::vector<int>> *text_line, float min_area);

 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::PSENetPostProcess> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    uint32_t imageWidth_ = 0;
    uint32_t imageHeight_ = 0;
};


#endif

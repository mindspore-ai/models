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

#ifndef HED_EDGEDETECTION_H
#define HED_EDGEDETECTION_H

#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

class HedEdgeDetection {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat *imageMat);
    APP_ERROR BinToTensorBase(const int length, float *image, const uint32_t target_channel,
                              const uint32_t target_width, const uint32_t target_height,
                              MxBase::TensorBase *tensorBase);
    APP_ERROR Inference(std::vector<MxBase::TensorBase> *inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR Process(const std::string &imgPath, const std::string &resultPath);
    APP_ERROR PostProcess(std::vector<MxBase::TensorBase> *inputs, float *buf);
    APP_ERROR readBin(std::string path, float *buf, int size);
    APP_ERROR writeBin(std::string path, float *buf, int size);
    int getBinSize(std::string path);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    uint32_t TargetChannel_ = 0;
    uint32_t TargetWidth_ = 0;
    uint32_t TargetHeight_ = 0;
};

#endif

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

#ifndef LIGHTCNN_H
#define LIGHTCNN_H
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

struct InitParam {
    uint32_t deviceId;
    bool softmax;
    bool checkTensor;
    std::string modelPath;
};


class Lightcnn {
 public:
    static const int IMG_H = 128;
    static const int IMG_W = 128;
    static const int FEATURE_NUM = 256;
    void getfilename(std::string *filename, const std::string &imgpath);
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat *img);
    APP_ERROR ResizeImage(const cv::Mat &srcImageMat,
                                     float (&imageArray)[IMG_H][IMG_W]);
    APP_ERROR ArrayToTensorBase(float (&imageArray)[IMG_H][IMG_W],
                              MxBase::TensorBase *tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR Process(const std::string &imgPath, const std::string &resultPath);

    double GetInferCostMilliSec() const { return inferCostTimeMilliSec; }

 private:
    APP_ERROR SaveResult(MxBase::TensorBase *tensor,
                       const std::string &resultpath);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    double inferCostTimeMilliSec = 0.0;
};

#endif

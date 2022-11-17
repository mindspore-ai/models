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

#ifndef POSENET_H
#define POSENET_H

#include <dirent.h>
#include <sys/stat.h>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    std::string dataset_name;
    bool checkTensor;
    std::string modelPath;
};

struct ImageShape {
    uint32_t width;
    uint32_t height;
};
std::vector<std::string> GetAllFiles(std::string dirName);
std::vector<std::string> GetAlldir(const std::string &dir_name,
                                   const std::string &data_name);
std::string RealPath(std::string path);
DIR *OpenDir(std::string dirName);

class Posenet {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
                        std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                          const std::map<std::string, float> &meta,
                          std::string &seq, std::string &image_file);
    APP_ERROR Process(const std::string &imgPath, const std::string &resultPath,
                      const std::string &dataset_name);
    APP_ERROR ReadImageCV(const std::string &imgPath, cv::Mat &imageMat,
                          ImageShape &imgShape);
    APP_ERROR ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat,
                          ImageShape &imgShape);
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat,
                                MxBase::TensorBase &tensorBase);
    APP_ERROR GetMetaMap(const ImageShape imgShape,
                         const ImageShape resizeimgShape,
                         std::map<std::string, float> &meta);
    APP_ERROR WriteResult(const std::string &imageFile,
                          std::vector<MxBase::TensorBase> &outputs,
                          const std::string &dataset_name,
                          const std::string &seq);

 private:
    APP_ERROR GetImageMeta(const ImageShape &imageShape,
                           MxBase::TensorBase &imgMetas) const;
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    double inferCostTimeMilliSec = 0.0;
};

#endif  // posenet_H

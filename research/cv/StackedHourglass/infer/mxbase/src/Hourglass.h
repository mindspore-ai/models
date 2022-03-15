/*
 *  Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MXBASE_HOURGLASS_H
#define MXBASE_HOURGLASS_H

#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <memory>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
  uint32_t deviceId;
  std::string modelPath;
  bool checkTensor;
};

float* readCFromFile(std::string path);
float* readSFromFile(std::string path);
float* readNFromFile(std::string path);
cv::Mat readMatFromFile(std::string path);
cv::Mat readGtFromFile(std::string path);
cv::Mat transform_preds(const cv::Mat &cropped_preds, cv::Mat &mat, const float c[], const float s);
void eval();

class Hourglass {
 public:
  APP_ERROR Init(const InitParam &initParam);
  APP_ERROR DeInit();
  APP_ERROR ReadInputTensor(const std::string &fileName, MxBase::TensorBase &tensorBase);
  APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
  APP_ERROR Process(const std::string &inferPath, const std::string &fileName);
  APP_ERROR WriteResult(const std::string &imageFile, std::vector<MxBase::TensorBase> &outputs);
  APP_ERROR ParseTensor(MxBase::TensorBase &tensors, MxBase::TensorBase &tensors1);
 private:
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
  MxBase::ModelDesc modelDesc_;
  uint32_t deviceId_ = 0;
  double inferCostTimeMilliSec = 0.0;
};

#endif

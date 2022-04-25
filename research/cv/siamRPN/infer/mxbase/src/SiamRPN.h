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

#ifndef FASTERRCNNPOST_FASTERRCNN_H
#define FASTERRCNNPOST_FASTERRCNN_H

#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <memory>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "PostProcess/SiamRPNMindsporePost.h"

struct InitParam {
  uint32_t deviceId;
  std::string labelPath;
  uint32_t classNum;
  float iouThresh;
  float scoreThresh;

  bool checkTensor;
  std::string modelPath;
};

std::vector<std::string> GetAlldir(const std::string &dir_name,
                                   const std::string &data_name);
DIR *OpenDir(std::string dirName);
class SiamRPN {
 public:
  APP_ERROR Init(const InitParam &initParam);
  APP_ERROR DeInit();

  APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat);

  APP_ERROR CVMatToTensorBase(cv::Mat &imageMat,
                              MxBase::TensorBase &tensorBase,
                              std::vector<uint32_t> &shape);
  APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
                      std::vector<MxBase::TensorBase> &outputs);
  APP_ERROR Process(const std::string &data_set,
                    const std::string &dataset_name);
  APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                        MxBase::Tracker &track, MxBase::Config &postConfig,
                        int idx, int total_num, float result_box[][4],
                        int &template_idx);
  void crop_and_pad(const cv::Mat& img, cv::Mat& dest, float cx, float cy, float original_sz, int im_h, int im_w);
  void Crop(const cv::Mat &img, cv::Mat &crop_img, const std::vector<int> &area);
  void Pad(const cv::Mat &srcImageMat, cv::Mat &dstImageMat, int left, int bottom, int right, int top);

 private:
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
  std::shared_ptr<MxBase::SiamRPNMindsporePost> post_;
  MxBase::ModelDesc modelDesc_;
  uint32_t deviceId_ = 0;
  double inferCostTimeMilliSec = 0.0;
};

#endif  // FASTERRCNNPOST_FASTERRCNN_H

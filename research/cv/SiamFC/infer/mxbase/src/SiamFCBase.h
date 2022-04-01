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

#ifndef MXBASE_SIAMFCBASE_H
#define MXBASE_SIAMFCBASE_H

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
extern std::vector<std::string> all_videos;

struct param {
  size_t size_s;
  double init_x;
  double init_y;
  double init_w;
  double init_h;
  double target_position[2];
  double target_sz[2];
  double wc_z;
  double hc_z;
  double s_z;
  double scale_z;
  double penalty[3] = {0.9745, 1, 0.9745};
  double scales[3] = {0.96385542, 1.00, 1.0375};
  std::string temp_video;
  std::string dataset_path;
  std::string dataset_path_txt;
  std::string record_name;
  std::string record_times;
  double s_x;
  double min_s_x;
  double max_s_x;
  double size_x_scales[3];
  std::vector<double> box;
  std::vector<std::string> all_files;
};
struct initParam {
  std::string modelPath1;
  std::string modelPath2;
  uint32_t deviceId;
};
const int DATA_LENTH = 1000;
class SiamFCBase {
 public:
  std::string RealPath(const std::string &path);
  APP_ERROR OpenDir(const std::string &dirName, DIR *&dir);
  APP_ERROR GetAllFiles(const std::string &dirName, const std::string &seq_name,
                        param &initParam);
  APP_ERROR GetPath(param &config, const std::string &temp_video,
                    int jogging_count, const std::string &seq_root_path,
                    const std::string &code_path);
  APP_ERROR Init(const initParam &init);
  APP_ERROR DeInit();
  APP_ERROR Inference(std::vector<MxBase::TensorBase> &input_exemplar,
                      MxBase::TensorBase instance,
                      std::vector<MxBase::TensorBase> &outputs_exemplar,
                      std::vector<MxBase::TensorBase> &output_exemplar);
  APP_ERROR Process(param &config);
  APP_ERROR PostProcess();
  APP_ERROR Getpos(const std::string &dirName, param &initParam);
  APP_ERROR CropAndPad(const cv::Mat &img, cv::Mat &target, float cx, float cy,
                       float size_z, float s_z);
  APP_ERROR HWC2CHW(const cv::Mat &srt, cv::Mat &dst, size_t resize_detection);
  APP_ERROR PreTreatMent(cv::Mat src, cv::Mat &target, param config, int size,
                         double s_x);
  APP_ERROR InitPosition(param &config, const std::string &temp_video);
  APP_ERROR GetSizeScales(param &config);
  APP_ERROR GetExemplar(const std::string &temp_video,
                        std::vector<MxBase::TensorBase> &outputs_exemplar,
                        param &config);
  APP_ERROR GetRetInstance(int instance_num, param &config,
                           std::vector<MxBase::TensorBase> &outputs_exemplar,
                           cv::Mat cos_window);
  APP_ERROR myCreateHanningWindow(cv::_OutputArray _dst, cv::Size winSize,
                                  int type);
  APP_ERROR CreMulHannWindow(cv::Size winSize, int type, cv::Mat &mulHanning);
  APP_ERROR CreateHanningWindowWithCV_64F(cv::Mat* _dst, int rows, int cols);
  APP_ERROR CreateHanningWindowWithCV_32F(cv::Mat* _dst, int rows, int cols);
  APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat,
                              MxBase::TensorBase &tensorBase);
  APP_ERROR TensorBaseToCVMat(const MxBase::TensorBase &tensorBase,
                              cv::Mat &imageMat,
                              const MxBase::ResizedImageInfo &resizedInfo);

 private:
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_1;
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_2;
  MxBase::ModelDesc modelDesc_1 = {};
  MxBase::ModelDesc modelDesc_2 = {};
  uint32_t deviceId_ = 0;
};
#endif

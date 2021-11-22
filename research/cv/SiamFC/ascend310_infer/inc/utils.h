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

#ifndef MINDSPORE_INFERENCE_UTILS_H_
#define MINDSPORE_INFERENCE_UTILS_H_
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/types_c.h>

#include <sys/stat.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <memory>
#include "include/api/types.h"

std::vector<std::string> GetAllFiles(const std::string_view& dirName,const std::string& seq_name);
DIR *OpenDir(const std::string_view& dirName);
std::string RealPath(const std::string_view& path);
mindspore::MSTensor ReadFileToTensor(const std::string &file);
int WriteResult(const std::string& imageFile, const std::vector<mindspore::MSTensor> &outputs);
cv::Mat BGRToRGB(cv::Mat img);
cv::Mat crop_and_pad(cv::Mat img, float cx, float cy, float size_z, float s_z);
std::vector<double> Getpos(const std::string &dirName);
float sumMat(cv::Mat& inputImg);
#endif

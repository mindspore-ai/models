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

#include "inc/utils.h"

#include <dirent.h>
#include <opencv2/imgproc/types_c.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

const int DAVID_DATA_SIZE = 471;
const int DAVID_DATA_BEGIN = 299;
const int DAVID_DATA_END = 770;
const int FOOTBALL_DATA_SIZE = 74;
const int FREEMAN3_DATA_SIZE = 460;
const int FREEMAN4_DATA_SIZE = 283;
const int DIVING_DATA_SIZE = 215;
using mindspore::DataType;
using mindspore::MSTensor;
using namespace std;

std::vector<std::string> GetAllFiles(const std::string_view& dirName,
                                     const std::string& seq_name) {
  struct dirent* filename;
  string seqName = string(dirName) + "/" + seq_name + "/img";

  DIR* dir = OpenDir(seqName);
  if (dir == nullptr) {
    cout << "no dir" << endl;
    return {};
  }
  std::vector<std::string> res;
  while ((filename = readdir(dir)) != nullptr) {
    std::string dName = std::string(filename->d_name);
    if (dName == "." || dName == ".." || filename->d_type != DT_REG) {
      continue;
    }
    res.emplace_back(string(dirName) + "/" + seq_name + "/img/" +
                     filename->d_name);
  }
  std::sort(res.begin(), res.end());
  std::vector<std::string> res_all;
  if (seq_name == "David") {
    res_all.resize(DAVID_DATA_SIZE);
    std::copy(res.begin() + DAVID_DATA_BEGIN, res.begin() + DAVID_DATA_END,
              res_all.begin());
  } else if (seq_name == "Football1") {
    res_all.resize(FOOTBALL_DATA_SIZE);
    std::copy(res.begin(), res.begin() + FOOTBALL_DATA_SIZE, res_all.begin());
  } else if (seq_name == "Freeman3") {
    res_all.resize(FREEMAN3_DATA_SIZE);
    std::copy(res.begin(), res.begin() + FREEMAN3_DATA_SIZE, res_all.begin());
  } else if (seq_name == "Freeman4") {
    res_all.resize(FREEMAN4_DATA_SIZE);
    std::copy(res.begin(), res.begin() + FREEMAN4_DATA_SIZE, res_all.begin());
  } else if (seq_name == "Diving") {
    res_all.resize(FREEMAN4_DATA_SIZE);
    std::copy(res.begin(), res.begin() + FREEMAN4_DATA_SIZE, res_all.begin());
  } else {
    for (size_t i = 0; i < res.size(); i++) {
      res_all.emplace_back(res[i]);
    }
  }
  return res_all;
}
std::vector<double> Getpos(const std::string& dirName) {
  std::ifstream infile;
  infile.open(dirName.c_str());
  std::string s;
  getline(infile, s);
  std::stringstream ss;
  ss << s;
  double temp;
  std::vector<double> data;
  while (ss >> temp) {
    data.push_back(temp);
    if (ss.peek() == ',' || ss.peek() == ' ' || ss.peek() == '\t') {
      ss.ignore();
    }
  }
  infile.close();
  return data;
}

int WriteResult(const std::string& imageFile,
                const std::vector<MSTensor>& outputs) {
  std::string homePath = "./result_Files";
  for (size_t i = 0; i < outputs.size(); ++i) {
    size_t outputSize;
    std::shared_ptr<const void> netOutput = outputs[i].Data();
    outputSize = outputs[i].DataSize();
    int pos = imageFile.rfind('/');
    std::string fileName(imageFile, pos + 1);
    fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'),
                     '_' + std::to_string(i) + ".bin");
    std::string outFileName = homePath + "/" + fileName;
    FILE* outputFile = fopen(outFileName.c_str(), "wb");
    fwrite(netOutput.get(), outputSize, sizeof(char), outputFile);
    fclose(outputFile);
    outputFile = nullptr;
  }
  return 0;
}

mindspore::MSTensor ReadFileToTensor(const std::string& file) {
  if (file.empty()) {
    std::cout << "Pointer file is nullptr" << std::endl;
    return mindspore::MSTensor();
  }

  std::ifstream ifs(file);
  if (!ifs.good()) {
    std::cout << "File: " << file << " is not exist" << std::endl;
    return mindspore::MSTensor();
  }

  if (!ifs.is_open()) {
    std::cout << "File: " << file << "open failed" << std::endl;
    return mindspore::MSTensor();
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  mindspore::MSTensor buffer(file, mindspore::DataType::kNumberTypeUInt8,
                             {static_cast<int64_t>(size)}, nullptr, size);

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char*>(buffer.MutableData()), size);
  ifs.close();

  return buffer;
}

DIR* OpenDir(const std::string_view& dirName) {
  if (dirName.empty()) {
    std::cout << " dirName is null ! " << std::endl;
    return nullptr;
  }
  std::string realPath = RealPath(dirName);
  struct stat s;
  lstat(realPath.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dirName is not a valid directory !" << std::endl;
    return nullptr;
  }
  DIR* dir = opendir(realPath.c_str());
  if (dir == nullptr) {
    std::cout << "Can not open dir " << dirName << std::endl;
    return nullptr;
  }
  std::cout << "Successfully opened the dir " << dirName << std::endl;
  return dir;
}

std::string RealPath(const std::string_view& path) {
  char realPathMem[PATH_MAX] = {0};
  char* realPathRet = nullptr;
  realPathRet = realpath(path.data(), realPathMem);
  if (realPathRet == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  std::string realPath(realPathMem);
  std::cout << path << " realpath is: " << realPath << std::endl;
  return realPath;
}

cv::Mat BGRToRGB(cv::Mat& img) {
  cv::Mat image(img.rows, img.cols, CV_8UC3);
  for (int i = 0; i < img.rows; ++i) {
    cv::Vec3b* p1 = img.ptr<cv::Vec3b>(i);
    cv::Vec3b* p2 = image.ptr<cv::Vec3b>(i);
    for (int j = 0; j < img.cols; ++j) {
      p2[j][2] = p1[j][0];
      p2[j][1] = p1[j][1];
      p2[j][0] = p1[j][2];
    }
  }
  return image;
}
cv::Mat crop_and_pad(cv::Mat img, float cx, float cy, float size_z, float s_z) {
  float xmin = cx - s_z / 2;
  float xmax = cx + s_z / 2;
  float ymin = cy - s_z / 2;
  float ymax = cy + s_z / 2;
  int w = img.cols;
  int h = img.rows;
  int left = 0;
  int right = 0;
  int top = 0;
  int bottom = 0;

  if (xmin < 0) left = static_cast<int>(abs(xmin));
  if (xmax > w) right = static_cast<int>(xmax - w);
  if (ymin < 0) top = static_cast<int>(abs(ymin));
  if (ymax > h) bottom = static_cast<int>(ymax - h);

  xmin = std::max(0, static_cast<int>(xmin));
  xmax = std::min(w, static_cast<int>(xmax));
  ymin = std::max(0, static_cast<int>(ymin));
  ymax = std::min(h, static_cast<int>(ymax));

  cv::Mat im_patch = img(cv::Range(ymin, ymax), cv::Range(xmin, xmax));
  if (left != 0 || right != 0 || top != 0 || bottom != 0) {
    cv::Scalar tempVal = cv::mean(img);
    tempVal.val[0] = static_cast<int>(tempVal.val[0]);
    tempVal.val[1] = static_cast<int>(tempVal.val[1]);
    tempVal.val[2] = static_cast<int>(tempVal.val[2]);
    cv::copyMakeBorder(im_patch, im_patch, top, bottom, left, right,
                       cv::BORDER_CONSTANT, tempVal);
  }
  if (size_z != s_z) {
    cv::resize(im_patch, im_patch, cv::Size(size_z, size_z));
  }
  return im_patch;
}

float sumMat(cv::Mat& inputImg) {
  float sum = 0.0;
  int rowNumber = inputImg.rows;
  int colNumber = inputImg.cols * inputImg.channels();
  for (int i = 0; i < rowNumber; i++) {
    uchar* data = inputImg.ptr<uchar>(i);
    for (int j = 0; j < colNumber; j++) {
      sum = data[j] + sum;
    }
  }

  return sum;
}

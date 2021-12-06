/*
 * Copyright 2021 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "Fairmot.h"

#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/property_tree/json_parser.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "acl/acl.h"
namespace {
const uint32_t YUV_BYTE_NU = 3;
const uint32_t YUV_BYTE_DE = 2;
const uint32_t FRAME_RATE = 25;
const uint32_t MODEL_HEIGHT = 768;
const uint32_t MODEL_WIDTH = 1280;
const float CONF_THRES = 0.4;
}  // namespace

void PrintTensorShape(const std::vector<MxBase::TensorDesc> &tensorDescVec,
                      const std::string &tensorName) {
  LogInfo << "The shape of " << tensorName << " is as follows:";
  for (size_t i = 0; i < tensorDescVec.size(); ++i) {
    LogInfo << "  Tensor " << i << ":";
    for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
      LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
    }
  }
}

APP_ERROR Fairmot::Init(const InitParam &initParam) {
  deviceId_ = initParam.deviceId;
  APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
  if (ret != APP_ERR_OK) {
    LogError << "Init devices failed, ret=" << ret << ".";
    return ret;
  }
  ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
  if (ret != APP_ERR_OK) {
    LogError << "Set context failed, ret=" << ret << ".";
    return ret;
  }
  dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
  ret = dvppWrapper_->Init();
  if (ret != APP_ERR_OK) {
    LogError << "DvppWrapper init failed, ret=" << ret << ".";
    return ret;
  }
  model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
  ret = model_->Init(initParam.modelPath, modelDesc_);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
    return ret;
  }

  PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
  PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");

  MxBase::ConfigData configData;
  const std::string checkTensor = initParam.checkTensor ? "true" : "false";

  configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
  configData.SetJsonValue("SCORE_THRESH",
                          std::to_string(initParam.scoreThresh));
  configData.SetJsonValue("IOU_THRESH", std::to_string(initParam.iouThresh));
  configData.SetJsonValue("CHECK_MODEL", checkTensor);

  auto jsonStr = configData.GetCfgJson().serialize();
  std::map<std::string, std::shared_ptr<void>> config;
  config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
  post_ = std::make_shared<MxBase::FairmotMindsporePost>();
  ret = post_->Init(config);
  if (ret != APP_ERR_OK) {
    LogError << "Fairmot init failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR Fairmot::DeInit() {
  dvppWrapper_->DeInit();
  model_->DeInit();
  post_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}

APP_ERROR Fairmot::Inference(const std::vector<MxBase::TensorBase> &inputs,
                             std::vector<MxBase::TensorBase> &outputs) {
  auto dtypes = model_->GetOutputDataType();
  for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
    std::vector<uint32_t> shape = {};
    for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
      shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
    }
    MxBase::TensorBase tensor(shape, dtypes[i],
                              MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                              deviceId_);
    APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
    if (ret != APP_ERR_OK) {
      LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
      return ret;
    }
    outputs.push_back(tensor);
  }
  MxBase::DynamicInfo dynamicInfo = {};
  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
  auto startTime = std::chrono::high_resolution_clock::now();
  APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
  auto endTime = std::chrono::high_resolution_clock::now();
  double costMs = std::chrono::duration<double, std::milli>(endTime - startTime)
                      .count();  // save time
  inferCostTimeMilliSec += costMs;
  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR Fairmot::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                               MxBase::JDETracker &tracker,
                               MxBase::Files &file) {
  APP_ERROR ret = post_->Process(inputs, tracker, file);
  if (ret != APP_ERR_OK) {
    LogError << "Process failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

void SaveInferResult(const std::vector<MxBase::ObjectInfo> &objInfos,
                     const std::string &resultPath) {
  if (objInfos.empty()) {
    LogWarn << "The predict result is empty.";
    return;
  }

  namespace pt = boost::property_tree;
  pt::ptree root, data;
  int index = 0;
  for (auto &obj : objInfos) {
    ++index;
    LogInfo << "BBox[" << index << "]:[x0=" << obj.x0 << ", y0=" << obj.y0
            << ", x1=" << obj.x1 << ", y1=" << obj.y1
            << "], confidence=" << obj.confidence << ", classId=" << obj.classId
            << ", className=" << obj.className << std::endl;
    pt::ptree item;
    item.put("classId", obj.classId);
    item.put("className", obj.className);
    item.put("confidence", obj.confidence);
    item.put("x0", obj.x0);
    item.put("y0", obj.y0);
    item.put("x1", obj.x1);
    item.put("y1", obj.y1);

    data.push_back(std::make_pair("", item));
  }
  root.add_child("data", data);
  pt::json_parser::write_json(resultPath, root, std::locale(), false);
}
std::string RealPath(std::string path) {
  char realPathMem[PATH_MAX] = {0};
  char *realPathRet = nullptr;
  realPathRet = realpath(path.data(), realPathMem);
  if (realPathRet == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  std::string realPath(realPathMem);
  std::cout << path << " realpath is: " << realPath << std::endl;
  return realPath;
}

DIR *OpenDir(std::string dirName) {
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
  DIR *dir = opendir(realPath.c_str());
  if (dir == nullptr) {
    std::cout << "Can not open dir " << dirName << std::endl;
    return nullptr;
  }
  std::cout << "Successfully opened the dir " << dirName << std::endl;
  return dir;
}

std::vector<std::string> GetAllFiles(std::string dirName) {
  struct dirent *filename;
  DIR *dir = OpenDir(dirName);
  if (dir == nullptr) {
    return {};
  }
  std::vector<std::string> res;
  while ((filename = readdir(dir)) != nullptr) {
    std::string dName = std::string(filename->d_name);
    if (dName == "." || dName == ".." || filename->d_type != DT_REG) {
      continue;
    }
    res.emplace_back(std::string(dirName) + "/" + filename->d_name);
  }
  std::sort(res.begin(), res.end());
  return res;
}

APP_ERROR Fairmot::ReadImageCV(const std::string &imgPath, cv::Mat &imageMat,
                               ImageShape &imgShape) {
  imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
  imgShape.width = imageMat.cols;
  imgShape.height = imageMat.rows;
  return APP_ERR_OK;
}

APP_ERROR Fairmot::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat,
                               ImageShape &imgShape) {
  int height = 608;
  int width = 1088;
  float ratio = std::min(static_cast<float>(height) / srcImageMat.rows,
                         static_cast<float>(width) / srcImageMat.cols);
  std::vector<int> new_shape = {
      static_cast<int>(round(srcImageMat.rows * ratio)),
      static_cast<int>(round(srcImageMat.cols * ratio))};
  int tmp = 2;
  float dw = static_cast<float>((width - new_shape[1])) / tmp;
  float dh = static_cast<float>((height - new_shape[0])) / tmp;
  int top = round(dh - 0.1);
  int bottom = round(dh + 0.1);
  int left = round(dw - 0.1);
  int right = round(dw + 0.1);
  cv::Mat tmp_img;
  cv::resize(srcImageMat, tmp_img, cv::Size(new_shape[1], new_shape[0]), 0, 0,
             cv::INTER_AREA);
  cv::Scalar value = cv::Scalar(127.5, 127.5, 127.5);
  cv::copyMakeBorder(tmp_img, dstImageMat, top, bottom, left, right,
                     cv::BORDER_CONSTANT, value);
  imgShape.width = dstImageMat.cols;
  imgShape.height = dstImageMat.rows;
  return APP_ERR_OK;
}

APP_ERROR Fairmot::CVMatToTensorBase(const cv::Mat &imageMat,
                                     MxBase::TensorBase &tensorBase) {
  const uint32_t dataSize = imageMat.cols * imageMat.rows * imageMat.channels();
  MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE,
                                   deviceId_);
  MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize,
                                   MxBase::MemoryData::MEMORY_HOST_MALLOC);

  APP_ERROR ret =
      MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc failed.";
    return ret;
  }
  std::vector<uint32_t> shape = {static_cast<uint32_t>(imageMat.rows),
                                 static_cast<uint32_t>(imageMat.cols),
                                 static_cast<uint32_t>(imageMat.channels())};
  tensorBase = MxBase::TensorBase(memoryDataDst, false, shape,
                                  MxBase::TENSOR_DTYPE_UINT8);
  return APP_ERR_OK;
}

APP_ERROR Fairmot::GetMetaMap(const ImageShape imgShape,
                              const ImageShape resizeimgShape,
                              MxBase::JDETracker &tracker) {
  std::vector<float> c = {static_cast<float>(imgShape.width) / 2,
                          static_cast<float>(imgShape.height) / 2};
  float s = std::max(static_cast<float>(resizeimgShape.width) /
                         static_cast<float>(resizeimgShape.height) *
                         static_cast<float>(imgShape.height),
                     static_cast<float>(imgShape.width)) *
            1.0;
  tracker.c = c;
  tracker.s = s;
  int tmp = 4;
  tracker.out_height = resizeimgShape.height / tmp;
  tracker.out_width = resizeimgShape.width / tmp;
  return APP_ERR_OK;
}

void Fairmot::WriteResult(const std::string &result_filename,
                          std::vector<MxBase::Results *> results) {
  FILE *fp;
  fp = std::fopen(result_filename.c_str(), "w");
  for (int i = 0; i < results.size(); i++) {
    std::vector<cv::Mat> online_tlwhs = (*results[i]).online_tlwhs;
    std::vector<int> online_ids = (*results[i]).online_ids;
    int frame_id = (*results[i]).frame_id;
    for (int j = 0; j < online_tlwhs.size(); j++) {
      if (online_ids[j] < 0) {
        continue;
      }
      double x1, y1, w, h;
      x1 = online_tlwhs[j].at<double>(0, 0);
      y1 = online_tlwhs[j].at<double>(0, 1);
      w = online_tlwhs[j].at<double>(0, 2);
      h = online_tlwhs[j].at<double>(0, 3);
      double x2, y2;
      x2 = x1 + w;
      y2 = y1 + h;
      fprintf(fp, "%d,%d,%.13lf,%.13lf,%.13lf,%.13lf,%d,%d,%d,%d\n", frame_id,
              (online_ids[j]), x1, y1, w, h, 1, -1, -1, -1);
    }
  }
  fclose(fp);
}

APP_ERROR Fairmot::Process(const std::string &imgPath) {
  ImageShape imageShape{};
  ImageShape resizedImageShape{};
  std::vector<std::string> seqs = {"/MOT20-01", "/MOT20-02", "/MOT20-03",
                                   "/MOT20-05"};
  std::string homePath = imgPath + "/result_Files";
  if (access(homePath.c_str(), 0) != 0) {
    std::string cmd = "mkdir " + homePath;
    system(cmd.data());
  }
  for (const auto &seq : seqs) {
    std::string result_filename = homePath + seq + ".txt";
    std::string image_path = imgPath + "/train" + seq + "/img1";
    std::vector<std::string> images = GetAllFiles(image_path);
    int frame_rate = 25;
    MxBase::JDETracker tracker(frame_rate);
    MxBase::Files file;
    for (const auto &image_file : images) {
      LogInfo << image_file;
      int tmp = 20;
      if (file.frame_id % tmp == 0) {
        LogInfo << "Processing frame " << file.frame_id;
      }
      cv::Mat imageMat;
      APP_ERROR ret = ReadImageCV(image_file, imageMat, imageShape);
      if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
      }
      ret = ResizeImage(imageMat, imageMat, resizedImageShape);
      if (ret != APP_ERR_OK) {
        LogError << "ResizeImage failed, ret=" << ret << ".";
        return ret;
      }
      ret = GetMetaMap(imageShape, resizedImageShape, tracker);
      if (ret != APP_ERR_OK) {
        LogError << "GetMetaMap failed, ret=" << ret << ".";
        return ret;
      }
      MxBase::TensorBase tensorBase;
      ret = CVMatToTensorBase(imageMat, tensorBase);
      if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
      }
      std::vector<MxBase::TensorBase> inputs = {};
      std::vector<MxBase::TensorBase> outputs = {};
      inputs.push_back(tensorBase);
      auto startTime = std::chrono::high_resolution_clock::now();
      ret = Inference(inputs, outputs);
      auto endTime = std::chrono::high_resolution_clock::now();
      double costMs =
          std::chrono::duration<double, std::milli>(endTime - startTime)
              .count();  // save time
      inferCostTimeMilliSec += costMs;
      if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
      }
      tracker.seq = seq;
      tracker.image_file = image_file;
      ret = PostProcess(outputs, tracker, file);
      if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
      }
    }
    WriteResult(result_filename, file.results);
  }
  return APP_ERR_OK;
}

/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd
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
#include <unistd.h>
#include <sys/stat.h>
#include "opencv2/opencv.hpp"
#include "ERFNetInfer/erfnetinfer.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

namespace {
  const uint32_t YUV_BYTE_NU = 3;
  const uint32_t YUV_BYTE_DE = 2;
  const uint32_t VPC_H_ALIGN = 2;
  const float CONFIDENCE = 0.5;
  const uint32_t MODEL_HEIGHT = 512;
  const uint32_t MODEL_WIDTH = 1024;
  const uint32_t GREEN = 255;

  const cv::Vec3b color_map[] = {
      cv::Vec3b(128, 64, 128),
      cv::Vec3b(244, 35, 232),
      cv::Vec3b(70, 70, 70),
      cv::Vec3b(102, 102, 156),
      cv::Vec3b(190, 153, 153),
      cv::Vec3b(153, 153, 153),
      cv::Vec3b(250, 170, 30),
      cv::Vec3b(220, 220, 0),
      cv::Vec3b(107, 142, 35),
      cv::Vec3b(152, 251, 152),
      cv::Vec3b(70, 130, 180),
      cv::Vec3b(220, 20, 60),
      cv::Vec3b(255, 0, 0),
      cv::Vec3b(0, 0, 142),
      cv::Vec3b(0, 0, 70),
      cv::Vec3b(0, 60, 100),
      cv::Vec3b(0, 80, 100),
      cv::Vec3b(0, 0, 230),
      cv::Vec3b(119, 11, 32),
      cv::Vec3b(0, 0, 0),
  };
}  // namespace

ERFNetInfer::ERFNetInfer(const uint32_t &deviceId, const std::string &modelPath) : deviceId_(deviceId) {
  APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
  if (ret != APP_ERR_OK) {
    LogError << "Init devices failed, ret=" << ret << ".";
    exit(-1);
  }
  ret = MxBase::TensorContext::GetInstance()->SetContext(deviceId_);
  if (ret != APP_ERR_OK) {
    LogError << "Set context failed, ret=" << ret << ".";
    exit(-1);
  }
  model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
  ret = model_->Init(modelPath, modelDesc_);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
    exit(-1);
  }
}

ERFNetInfer::~ERFNetInfer() {
  model_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
}

APP_ERROR ERFNetInfer::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
  const uint32_t dataSize = imageMat.cols * imageMat.rows * 3 * 4;
  MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);

  static unsigned char data[3 * 512 * 1024 * 4];
  for (size_t h = 0; h < 512; h++)
    for (size_t w = 0; w < 1024; w++)
      for (size_t c = 0; c < 3; c++) {
        data[(c * 1024 * 512 + h * 1024 + w)*4 + 0] = imageMat.data[(h * 1024 * 3 + w * 3 + c)*4 + 0];
        data[(c * 1024 * 512 + h * 1024 + w)*4 + 1] = imageMat.data[(h * 1024 * 3 + w * 3 + c)*4 + 1];
        data[(c * 1024 * 512 + h * 1024 + w)*4 + 2] = imageMat.data[(h * 1024 * 3 + w * 3 + c)*4 + 2];
        data[(c * 1024 * 512 + h * 1024 + w)*4 + 3] = imageMat.data[(h * 1024 * 3 + w * 3 + c)*4 + 3];
      }
  MxBase::MemoryData memoryDataSrc(data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

  APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc failed.";
    return ret;
  }
  std::vector<uint32_t> shape = {1, 3, static_cast<uint32_t>(imageMat.rows), static_cast<uint32_t>(imageMat.cols)};
  *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
  return APP_ERR_OK;
}

APP_ERROR ERFNetInfer::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                 std::vector<MxBase::TensorBase> *outputs) {
  auto dtypes = model_->GetOutputDataType();
  for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
    std::vector<uint32_t> shape = {};
    for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
      shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
    }
    MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
    if (ret != APP_ERR_OK) {
      LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
      return ret;
    }
    outputs->push_back(tensor);
  }
  MxBase::DynamicInfo dynamicInfo = {};
  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;

  APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR ERFNetInfer::WriteResult(MxBase::TensorBase *tensor,
                                   const std::string &imgPath) {
  APP_ERROR ret = tensor->ToHost();
  if (ret != APP_ERR_OK) {
    LogError << "ToHost failed.";
    return ret;
  }
  cv::Mat imgrgb = cv::Mat(512, 1024, CV_8UC3);

  // 1 x 20 x 512 x 1024
  auto data = reinterpret_cast<float *>(tensor->GetBuffer());
  float inferPixel[20];
  for (size_t x = 0; x < 512; ++x) {
    for (size_t y = 0; y < 1024; ++y) {
      for (size_t c = 0; c < 20; ++c) {
        inferPixel[c] = *(data + c * 1024 * 512 + x * 1024 + y);  // c, x, y
      }
      size_t max_index = std::max_element(inferPixel, inferPixel + 20) - inferPixel;
      imgrgb.at<cv::Vec3b>(x, y) = color_map[max_index];
    }
  }
  cv::imwrite(imgPath, imgrgb);
  return APP_ERR_OK;
}

APP_ERROR ERFNetInfer::Process(const std::string &imgPath, const std::string &output_path) {
  const std::string img_name(imgPath.begin() + imgPath.find_last_of('/'), imgPath.end());
  const std::string output_img_path = output_path + img_name;

  cv::Mat imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
  cv::resize(imageMat, imageMat, cv::Size(1024, 512));
  cv::cvtColor(imageMat, imageMat, cv::COLOR_RGB2BGR);
  imageMat.convertTo(imageMat, CV_32F);
  imageMat = imageMat / 255;

  MxBase::TensorBase tensorBase;
  APP_ERROR ret = CVMatToTensorBase(imageMat, &tensorBase);

  if (ret != APP_ERR_OK) {
    LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
    return ret;
  }

  std::vector<MxBase::TensorBase> inputs = {};
  std::vector<MxBase::TensorBase> outputs = {};
  inputs.push_back(tensorBase);

  ret = Inference(inputs, &outputs);
  if (ret != APP_ERR_OK) {
    LogError << "Inference failed, ret=" << ret << ".";
    return ret;
  }

  ret = WriteResult(&outputs[0], output_img_path);
  if (ret != APP_ERR_OK) {
    LogError << "Save result failed, ret=" << ret << ".";
    return ret;
  }

  return APP_ERR_OK;
}

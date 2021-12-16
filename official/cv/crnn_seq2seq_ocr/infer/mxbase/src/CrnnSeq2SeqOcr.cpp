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
#include "CrnnSeq2SeqOcr.h"
#include <memory>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

namespace {
const uint32_t YUV_BYTE_NU = 3;
const uint32_t YUV_BYTE_DE = 2;
const uint32_t VPC_H_ALIGN = 2;
}

APP_ERROR CrnnSeqToSeqOcr::Init(const InitParam &initParam) {
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
  model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
  ret = model_->Init(initParam.modelPath, modelDesc_);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
    return ret;
  }

  return APP_ERR_OK;
}

APP_ERROR CrnnSeqToSeqOcr::DeInit() {
  model_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}

cv::Mat hwc2chw(const cv::Mat &image) {
  std::vector<cv::Mat> rgb_images;
  cv::split(image, rgb_images);

  // Stretch one-channel images to vector
  cv::Mat m_flat_r = rgb_images[0].reshape(1, 1);
  cv::Mat m_flat_g = rgb_images[1].reshape(1, 1);
  cv::Mat m_flat_b = rgb_images[2].reshape(1, 1);

  // Now we can rearrange channels if need
  cv::Mat matArray[] = {m_flat_r, m_flat_g, m_flat_b};

  cv::Mat flat_image;
  // Concatenate three vectors to one
  cv::hconcat(matArray, 3, flat_image);
  return flat_image;
}

APP_ERROR CrnnSeqToSeqOcr::ReadAndResize(const std::string &imgPath,
                                         MxBase::TensorBase *outputTensor) {
  cv::Mat srcImageMat = cv::imread(imgPath);
  cv::Mat dstImageMat;
  uint32_t resizeWidth = 512;
  uint32_t resizeHeight = 128;

  cv::resize(srcImageMat, srcImageMat, cv::Size(resizeWidth, resizeHeight));
  srcImageMat.convertTo(srcImageMat, CV_32F);
  cv::normalize(srcImageMat, srcImageMat, -1, 1, cv::NORM_MINMAX);
  dstImageMat = hwc2chw(srcImageMat);
  uint32_t dataSize = srcImageMat.cols * srcImageMat.rows * 3 * 4;
  MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
  MxBase::MemoryData memoryDataSrc(dstImageMat.data, dataSize,
                           MxBase::MemoryData::MEMORY_HOST_MALLOC);

  APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc failed.";
    return ret;
  }

  std::vector<uint32_t> shape = {eval_batch_size, 3,
                                 static_cast<uint32_t>(srcImageMat.rows),
                                 static_cast<uint32_t>(srcImageMat.cols)};
  *outputTensor = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
  return APP_ERR_OK;
}

APP_ERROR
CrnnSeqToSeqOcr::Inference(const std::vector<MxBase::TensorBase> &inputs,
                           std::vector<MxBase::TensorBase> *outputs) {
  auto dtypes = model_->GetOutputDataType();
  for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
    std::vector<uint32_t> shape = {};
    for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
      shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
    }

    MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                      deviceId_);
    APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
    if (ret != APP_ERR_OK) {
      LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
      return ret;
    }

    outputs->push_back(tensor);
  }

  MxBase::DynamicInfo dynamicInfo = {};
  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;

  auto startTime = std::chrono::high_resolution_clock::now();
  APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
  auto endTime = std::chrono::high_resolution_clock::now();
  double costMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();
  inferCostTimeMilliSec += costMs;

  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }

  return APP_ERR_OK;
}

APP_ERROR CrnnSeqToSeqOcr::PostProcess(std::vector<MxBase::TensorBase> *tensors,
                                       std::string *textInfos) {
  int32_t ans;
  for (auto &it : *tensors) {
    it.ToHost();
    it.GetValue(ans, 0);
    if (ans == eos_id_) {
      break;
    }
    *textInfos = *textInfos + " " + std::to_string(ans);
  }
  return APP_ERR_OK;
}

APP_ERROR MakeInput(std::vector<uint32_t> shape, void *data,
                    const MxBase::TensorDataType &type, uint32_t deviceId_,
                    MxBase::TensorBase *input1, uint32_t rate) {
  uint32_t dataSize = std::accumulate(shape.begin(), shape.end(), rate);
  MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
  MxBase::MemoryData memoryDataSrc(data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
  APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc failed.";
    return ret;
  }
  *input1 = MxBase::TensorBase(memoryDataDst, false, shape, type);
  return APP_ERR_OK;
}

APP_ERROR CrnnSeqToSeqOcr::Process(const std::string &imgPath,
                                   std::string *result) {
  int32_t *data1 = reinterpret_cast<int32_t *>(malloc(eval_batch_size * sizeof(int32_t)));
  uint16_t *data2 =
      reinterpret_cast<uint16_t *>(malloc(eval_batch_size * decoder_hidden_size * sizeof(uint16_t)));
  for (uint32_t i = 0; i < eval_batch_size; i++) {
    data1[i] = 1;
  }
  for (uint32_t i = 0; i < eval_batch_size * decoder_hidden_size; i++) {
    data2[i] = 0;
  }
  MxBase::TensorBase resizeImage;
  APP_ERROR ret = ReadAndResize(imgPath, &resizeImage);
  if (ret != APP_ERR_OK) {
    LogError << "Read and resize image failed, ret=" << ret << ".";
    return ret;
  }
  MxBase::TensorBase input1, input2;
  std::vector<uint32_t> shape1 = {1, eval_batch_size};
  std::vector<uint32_t> shape2 = {1, eval_batch_size, decoder_hidden_size};
  MakeInput(shape1, data1, MxBase::TENSOR_DTYPE_INT32, deviceId_, &input1, 4);
  MakeInput(shape2, data2, MxBase::TENSOR_DTYPE_FLOAT16, deviceId_, &input2, 2);
  std::vector<MxBase::TensorBase> inputs = {};
  std::vector<MxBase::TensorBase> outputs = {};
  inputs.push_back(resizeImage);
  inputs.push_back(input1);
  inputs.push_back(input2);

  ret = Inference(inputs, &outputs);
  free(data1);
  free(data2);
  if (ret != APP_ERR_OK) {
    LogError << "Inference failed, ret=" << ret << ".";
    return ret;
  }

  ret = PostProcess(&outputs, result);
  if (ret != APP_ERR_OK) {
    LogError << "PostProcess failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

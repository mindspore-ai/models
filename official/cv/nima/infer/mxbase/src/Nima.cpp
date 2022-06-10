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

#include "Nima.h"
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include <iostream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include <opencv2/dnn.hpp>
#include "MxBase/Log/Log.h"

using MxBase::DeviceManager;
using MxBase::TensorBase;
using MxBase::MemoryData;
using  namespace MxBase;

APP_ERROR Nima::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = DeviceManager::GetInstance()->InitDevices();
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

APP_ERROR Nima::DeInit() {
    model_->DeInit();
    DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Nima::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);
    return APP_ERR_OK;
}

APP_ERROR Nima::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    // resize image to 224 * 224
    static constexpr uint32_t resizeHeight = 224;
    static constexpr uint32_t resizeWidth = 224;
    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR Nima::hwc_to_chw(const cv::Mat &dstImageMat, float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH]) {
    // HWC->CHW
    for (int i = 0; i < NET_HEIGHT; i++) {
        for (int j = 0; j < NET_WIDTH; j++) {
            cv::Vec3b nums = dstImageMat.at<cv::Vec3b>(i, j);
            imageArray[0][i][j] = nums[0];
            imageArray[1][i][j] = nums[1];
            imageArray[2][i][j] = nums[2];
        }
    }
    return APP_ERR_OK;
}

APP_ERROR Nima::NormalizeImage(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH], \
                                        float (&imageArrayNormal)[CHANNEL][NET_HEIGHT][NET_WIDTH]) {
    float mean[] = {0.485, 0.456, 0.406};
    float std[] = {0.229, 0.224, 0.225};
    for (int i = 0; i < CHANNEL; i++) {
        for (int j = 0; j < NET_HEIGHT; j++) {
            for (int k = 0; k < NET_WIDTH; k++) {
                imageArrayNormal[i][j][k] = (imageArray[i][j][k] - mean[i]) / std[i];
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR Nima::ArrayToTensorBase(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH], \
                                           MxBase::TensorBase* tensorBase) {
    const uint32_t dataSize = CHANNEL * NET_HEIGHT * NET_WIDTH * sizeof(float);
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageArray, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {CHANNEL, NET_HEIGHT, NET_WIDTH};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR Nima::Inference(std::vector<TensorBase> &inputs, std::vector<TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
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
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "Inference success";
    return APP_ERR_OK;
}

APP_ERROR  Nima::WriteResult(const std::string &imageFile, std::vector<MxBase::TensorBase> &outputs) {
  std::string infer_result_name = "../result" + std::string("/test.txt");
  LogInfo << "infer_result_name:" << infer_result_name;
  LogInfo << "image File：" << imageFile;
  std::string imageName = imageFile.substr(imageFile.find_last_of("/") + 1);
  LogInfo << "image Name：" << imageName;
  for (auto tensor : outputs) {
    LogInfo << "tensor size is:" << tensor.GetSize() <<  std::endl;
    LogInfo << "tensor detailed information is:" << tensor.GetDesc() << std::endl;
    auto out_data = reinterpret_cast<const float *>(tensor.GetBuffer());
    LogInfo << "output data is:";
    std::ofstream file_stream1(infer_result_name.c_str(), std::ios::app);
    file_stream1 << imageName << ":";
    for (int j = 0; j < 10; j++) {
      std::stringstream out_data_1;
      out_data_1 << out_data[j] << " ";
      std::cout << out_data[j] << " ";
      file_stream1 << out_data_1.str();
    }
    file_stream1 << "\n";
    file_stream1.close();
    std::cout << std::endl;
  }
  return APP_ERR_OK;
}

APP_ERROR Nima::Process(const std::string &imgPath) {
    std::vector<TensorBase> inputs = {};
    std::vector<TensorBase> outputs = {};

    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    // pre-process
    // step 1
    cv::Mat dstImageMat;
    ResizeImage(imageMat, dstImageMat);
    // step 2
    float img_array[CHANNEL][NET_HEIGHT][NET_WIDTH];
    hwc_to_chw(dstImageMat, img_array);
    // step 3
    float img_normalize[CHANNEL][NET_HEIGHT][NET_WIDTH];
    NormalizeImage(img_array, img_normalize);
    TensorBase tensorBase;
    ret = ArrayToTensorBase(img_normalize, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "Array to TensorBase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);

    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        ret = outputs[i].ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "to host fail.";
            return ret;
        }
        auto *netOutput = reinterpret_cast<float *>(outputs[i].GetBuffer());
        std::vector<uint32_t> out_shape = outputs[i].GetShape();
    }

    ret = WriteResult(imgPath, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

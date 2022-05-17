/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "Nasfpn.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <queue>
#include <utility>
#include <fstream>
#include <map>
#include <iostream>
#include "acl/acl.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

namespace {
    uint16_t RESIZE_IMAGE_HEIGHT = 640;
    uint16_t RESIZE_IMAGE_WIDTH = 640;
    uint16_t FLOAT_SIZE = 4;
    const std::vector<uint32_t> length = {4, 81};
    const uint32_t BOX_NUM = 76725;
    const std::vector<float> means = {123.675, 116.28, 103.53};
    const std::vector<float> stds = {58.395, 57.12, 57.375};
}  // namespace

int WriteResult(const std::string& imageFile, const std::vector<MxBase::TensorBase> &outputs) {
    std::string homePath = "./result";
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        outputSize = outputs[i].GetSize();
        std::string outFileName = homePath + "/" + imageFile + '_' + std::to_string(i) + ".bin";
        float *boxes = reinterpret_cast<float *>(outputs[i].GetBuffer());
        FILE * outputFile = fopen(outFileName.c_str(), "wb");
        fwrite(boxes, sizeof(float), length[i] * BOX_NUM, outputFile);
        fclose(outputFile);
        outputFile = nullptr;
    }
    return 0;
}

APP_ERROR nasfpn::Init(const InitParam &initParam) {
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
    model_nasfpn = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_nasfpn->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR nasfpn::DeInit() {
    model_nasfpn->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR nasfpn::ReadImage(const std::string &imgPath, std::vector<std::vector<std::vector<float>>> &im) {
    cv::Mat imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::resize(imageMat, imageMat, cv::Size(RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH));

    std::vector<float> rgb;
    for (int i = 0; i < RESIZE_IMAGE_HEIGHT; i++) {
        std::vector<std::vector<float>> t;
        im.push_back(t);
        for (int j = 0; j < RESIZE_IMAGE_WIDTH; j++) {
            float r = (imageMat.at<cv::Vec3b>(i, j)[2] - means[0]) /stds[0];
            float g =  (imageMat.at<cv::Vec3b>(i, j)[1] - means[1]) /stds[1];
            float b =  (imageMat.at<cv::Vec3b>(i, j)[0] - means[2]) /stds[2];
            rgb = {r, g, b};

            im[i].push_back(rgb);
        }
    }

    return APP_ERR_OK;
}

APP_ERROR nasfpn::VectorToTensorBase_float(const std::vector<std::vector<std::vector<float>>> &batchFeatureVector,
                                    MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 1;
    std::vector<uint32_t> shape = {};
    shape.push_back(1);
    shape.push_back(batchFeatureVector[0][0].size());
    shape.push_back(batchFeatureVector.size());
    shape.push_back(batchFeatureVector[0].size());

    for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    float *metaFeatureData = new float[dataSize];
    uint32_t idx = 0;
    for (size_t ch = 0; ch < batchFeatureVector[0][0].size(); ch++) {
        for (size_t h = 0; h < batchFeatureVector.size(); h++) {
              for (size_t w = 0; w < batchFeatureVector[0].size(); w++) {
                metaFeatureData[idx++] = batchFeatureVector[h][w][ch];
                }
        }
    }
    MxBase::MemoryData memoryDataDst(dataSize * FLOAT_SIZE, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(metaFeatureData, dataSize * FLOAT_SIZE, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}


APP_ERROR nasfpn::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_nasfpn->GetOutputDataType();
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
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_nasfpn->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference nasfpn failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR nasfpn::Process(const std::string& imageFile, const std::string &image_path, const InitParam &initParam) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs_tb = {};


    std::vector<std::vector<std::vector<float>>> image;
    ReadImage(image_path, image);
    MxBase::TensorBase tensorBase;
    APP_ERROR ret = VectorToTensorBase_float(image, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret1 = Inference(inputs, outputs_tb);

    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret1 != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret1 << ".";
        return ret1;
    }

    if (!outputs_tb[0].IsHost()) {
            outputs_tb[0].ToHost();
        }

    if (!outputs_tb[1].IsHost()) {
            outputs_tb[1].ToHost();
        }

    WriteResult(imageFile, outputs_tb);
}

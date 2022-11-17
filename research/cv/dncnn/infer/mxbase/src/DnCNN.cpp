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

#include "DnCNN.h"
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
    const int FLOAT_SIZE = 4;
}

void WriteResult(const int index, const std::vector<MxBase::TensorBase> &outputs) {
    std::string homePath = "./result";
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        outputSize = outputs[i].GetSize();
        std::string outFileName = homePath + "/output_" + std::to_string(index) + ".bin";
        float *boxes = reinterpret_cast<float *>(outputs[i].GetBuffer());
        FILE * outputFile = fopen(outFileName.c_str(), "wb");
        fwrite(boxes, sizeof(float), 256 * 256, outputFile);
        fclose(outputFile);
        outputFile = nullptr;
    }
}

APP_ERROR DnCNN::Init(const InitParam &initParam) {
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
    model_DnCNN = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_DnCNN->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR DnCNN::DeInit() {
    model_DnCNN->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR DnCNN::ReadBin(const std::string &path, std::vector<std::vector<float>> &dataset) {
    std::ifstream inFile(path, std::ios::binary);
    float data[256*256];
    inFile.read(reinterpret_cast<char *>(&data), sizeof(data));
    std::vector<float> temp(data, data+sizeof(data) / sizeof(data[0]));
    dataset.push_back(temp);

    return APP_ERR_OK;
}


APP_ERROR DnCNN::VectorToTensorBase(const std::vector<std::vector<float>> &input,
                                          MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 4*1*256*256;
    float *metaFeatureData = new float[dataSize];
    uint32_t idx = 0;
    for (size_t bs = 0; bs < input.size(); bs++) {
        for (size_t c = 0; c < input[bs].size(); c++) {
            metaFeatureData[idx++] = input[bs][c];
        }
    }
    MxBase::MemoryData memoryDataDst(dataSize * FLOAT_SIZE, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(metaFeatureData), dataSize * FLOAT_SIZE,
                                     MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, 1, 256, 256};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);

    return APP_ERR_OK;
}


APP_ERROR DnCNN::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_DnCNN->GetOutputDataType();
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
    APP_ERROR ret = model_DnCNN->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference DnCNN failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR DnCNN::Process(const int index, const std::string &image_path,
                         const InitParam &initParam, std::vector<int> &outputs) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs_tb = {};

    std::vector<std::vector<float>> image_data;
    APP_ERROR ret = ReadBin(image_path, image_data);

    if (ret != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    MxBase::TensorBase tensorBase;
    APP_ERROR ret1 = VectorToTensorBase(image_data, tensorBase);
    if (ret1 != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret1 << ".";
        return ret1;
    }

    inputs.push_back(tensorBase);
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret3 = Inference(inputs, outputs_tb);

    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret3 != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret3 << ".";
        return ret3;
    }

    if (!outputs_tb[0].IsHost()) {
            outputs_tb[0].ToHost();
        }
    WriteResult(index, outputs_tb);
}

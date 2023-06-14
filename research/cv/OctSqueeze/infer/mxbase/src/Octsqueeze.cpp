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

#include "Octsqueeze.h"
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
    uint32_t FLOAT_SIZE = 4;
    uint32_t MAX_LENGTH = 1000;
}  // namespace

APP_ERROR octsqueeze::Init(const InitParam &initParam) {
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
    model_octsqueeze = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_octsqueeze->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR octsqueeze::DeInit() {
    model_octsqueeze->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR octsqueeze::VectorToTensorBase_float32(std::vector<std::vector<float>> &batchFeatureVector,
                                    MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 1;
    std::vector<uint32_t> shape = {};
    shape.push_back(1);
    shape.push_back(1);
    shape.push_back(MAX_LENGTH);
    shape.push_back(batchFeatureVector[0].size());

    for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    float *metaFeatureData = new float[dataSize];
    uint32_t idx = 0;

    for (size_t w = 0; w < batchFeatureVector.size(); w++) {
      for (size_t h = 0; h < batchFeatureVector[0].size(); h++) {
        metaFeatureData[idx++] = batchFeatureVector[w][h];
      }
    }

    if (batchFeatureVector.size() < MAX_LENGTH) {
      for (size_t w = 0; w < MAX_LENGTH - batchFeatureVector.size(); w++) {
        for (size_t h = 0; h < batchFeatureVector[0].size(); h++) {
            metaFeatureData[idx++] = 0.0;
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


APP_ERROR octsqueeze::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_octsqueeze->GetOutputDataType();
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
    APP_ERROR ret = model_octsqueeze->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference octsqueeze failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR octsqueeze::Process(std::vector<std::vector<float>> &input, const InitParam &initParam,
                          float* &outputs) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs_tb = {};

    MxBase::TensorBase tensorBase;

    APP_ERROR ret = VectorToTensorBase_float32(input, tensorBase);
    input.clear();
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
    outputs = reinterpret_cast<float *>(outputs_tb[0].GetBuffer());
}

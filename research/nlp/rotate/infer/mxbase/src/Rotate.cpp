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

#include "Rotate.h"
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
    uint16_t UINT32_SIZE = 4;
}  // namespace

APP_ERROR rotate::Init(const InitParam &initParam) {
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
    model_rotate = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_rotate->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR rotate::DeInit() {
    model_rotate->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR rotate::VectorToTensorBase_int32(const std::vector<uint32_t> &batchFeatureVector,
                                    MxBase::TensorBase  &tensorBase) {
    uint32_t dataSize = 1;
    std::vector<uint32_t> shape = {};
    shape.push_back(1);
    shape.push_back(batchFeatureVector.size());

    for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    uint32_t *metaFeatureData = new uint32_t[dataSize];
    uint32_t idx = 0;

    for (size_t w = 0; w < batchFeatureVector.size(); w++) {
      metaFeatureData[idx++] = batchFeatureVector[w];
    }
    MxBase::MemoryData memoryDataDst(dataSize * UINT32_SIZE, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(metaFeatureData, dataSize * UINT32_SIZE, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_INT32);
    return APP_ERR_OK;
}


APP_ERROR rotate::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase>  &outputs) {
    auto dtypes = model_rotate->GetOutputDataType();
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
    APP_ERROR ret = model_rotate->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference rotate failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR rotate::Process(const std::vector<uint32_t> &positive_sample, const std::vector<uint32_t> &negative_sample,
                          const std::vector<uint32_t> &filter_bias, const InitParam &initParam,
                          std::vector<uint32_t>  &argsort_o, uint32_t positive_arg_o) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs_tb = {};
    std::string argsort_path = "./results/argsort.txt";
    std::string positive_arg_path = "./results/positive_arg.txt";

    MxBase::TensorBase tensorBase1;
    MxBase::TensorBase tensorBase2;
    MxBase::TensorBase tensorBase3;

    APP_ERROR ret = VectorToTensorBase_int32(positive_sample, tensorBase1);
    if (ret != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    APP_ERROR ret1 = VectorToTensorBase_int32(negative_sample, tensorBase2);
    if (ret1 != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret1 << ".";
        return ret1;
    }

    APP_ERROR ret2 = VectorToTensorBase_int32(filter_bias, tensorBase3);
    if (ret2 != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret2 << ".";
        return ret2;
    }

    inputs.push_back(tensorBase1);
    inputs.push_back(tensorBase2);
    inputs.push_back(tensorBase3);

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
    uint32_t *argsort = reinterpret_cast<uint32_t *>(outputs_tb[0].GetBuffer());

    if (!outputs_tb[1].IsHost()) {
            outputs_tb[1].ToHost();
        }
    uint32_t *positive_arg = reinterpret_cast<uint32_t *>(outputs_tb[1].GetBuffer());

    std::ofstream outfile1(argsort_path, std::ios::app);
    std::ofstream outfile2(positive_arg_path, std::ios::app);

    if (outfile1.fail() && outfile2.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }

    uint32_t length = initParam.seq_len;
    for (int n = 0; n < length; n++) {
        outfile1 << argsort[n];
        if (n == length - 1) {
            outfile1<< "\n";
        } else {
            outfile1<< "\t";
        }
        argsort_o.push_back(argsort[n]);
    }

    positive_arg_o = positive_arg[0];
    outfile2 << positive_arg_o << "\n";

    outfile1.close();
    outfile2.close();
}


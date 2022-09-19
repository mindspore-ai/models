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

#include "Tbnet.h"
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
    const std::vector<std::vector<uint32_t>> SHAPE = {{1}, {1, 39}, {1, 39},
                                                      {1, 39}, {1, 39}, {1}};
    const int FLOAT_SIZE = 4;
    const int INT_SIZE = 8;
    const int DATA_SIZE_1 = 1;
    const int DATA_SIZE_39 = 39;
}

void WriteResult(const std::string &file_name, const std::vector<MxBase::TensorBase> &outputs) {
    std::string homePath = "./result";
    for (size_t i = 0; i < outputs.size(); ++i) {
        float *boxes = reinterpret_cast<float *>(outputs[i].GetBuffer());
        std::string outFileName = homePath + "/tbnet_item_bs1_" + file_name + "_" +
                                  std::to_string(i) + ".txt";
        std::ofstream outfile(outFileName, std::ios::app);
        size_t outputSize;
        outputSize = outputs[i].GetSize();
        for (size_t j = 0; j < outputSize; ++j) {
            if (j != 0) {
                outfile << ",";
            }
            outfile << boxes[j];
        }
        outfile.close();
    }
}

APP_ERROR Tbnet::Init(const InitParam &initParam) {
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
    model_Tbnet = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_Tbnet->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR Tbnet::DeInit() {
    model_Tbnet->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR Tbnet::ReadBin_float(const std::string &path, std::vector<std::vector<float>> &dataset,
                         const int datasize) {
    std::ifstream inFile(path, std::ios::binary);

    float *data = new float[datasize];
    inFile.read(reinterpret_cast<char *>(data), datasize * sizeof(data[0]));
    std::vector<float> temp(data, data + datasize);
    dataset.push_back(temp);

    return APP_ERR_OK;
}

APP_ERROR Tbnet::ReadBin_int(const std::string &path, std::vector<std::vector<int64_t>> &dataset,
                         const int datasize) {
    std::ifstream inFile(path, std::ios::binary);

    int64_t *data = new int64_t[datasize];
    inFile.read(reinterpret_cast<char *>(data), datasize * sizeof(data[0]));
    std::vector<int64_t> temp(data, data + datasize);
    dataset.push_back(temp);

    return APP_ERR_OK;
}

APP_ERROR Tbnet::VectorToTensorBase_float(const std::vector<std::vector<float>> &input,
                                    MxBase::TensorBase &tensorBase,
                                    const std::vector<uint32_t> &shape) {
    uint32_t dataSize = 1;
    for (int i = 0; i < shape.size(); i++) {
        dataSize = dataSize * shape[i];
    }     // input shape
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

    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);

    return APP_ERR_OK;
}

APP_ERROR Tbnet::VectorToTensorBase_int(const std::vector<std::vector<int64_t>> &input,
                                    MxBase::TensorBase &tensorBase,
                                    const std::vector<uint32_t> &shape) {
    int dataSize = 1;
    for (int i = 0; i < shape.size(); i++) {
        dataSize = dataSize * shape[i];
    }     // input shape

    int64_t *metaFeatureData = new int64_t[dataSize];

    uint32_t idx = 0;
    for (size_t bs = 0; bs < input.size(); bs++) {
        for (size_t c = 0; c < input[bs].size(); c++) {
            metaFeatureData[idx++] = input[bs][c];
        }
    }
    MxBase::MemoryData memoryDataDst(dataSize * INT_SIZE, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(metaFeatureData), dataSize * INT_SIZE,
                                     MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_INT64);

    return APP_ERR_OK;
}

APP_ERROR Tbnet::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_Tbnet->GetOutputDataType();
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
    APP_ERROR ret = model_Tbnet->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference Tbnet failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Tbnet::Process(const int &index, const std::string &datapath,
                         const InitParam &initParam, std::vector<int> &outputs) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs_tb = {};

    std::vector<std::vector<int64_t>> item;
    APP_ERROR ret = ReadBin_int(datapath + "00_item/tbnet_item_bs1_" +
                            std::to_string(index) + ".bin", item, DATA_SIZE_1);
    std::vector<std::vector<int64_t>> rl1;
    ReadBin_int(datapath + "01_rl1/tbnet_rl1_bs1_" +
                std::to_string(index) + ".bin", rl1, DATA_SIZE_39);
    std::vector<std::vector<int64_t>> ety;
    ReadBin_int(datapath + "02_ety/tbnet_ety_bs1_" +
                std::to_string(index) + ".bin", ety, DATA_SIZE_39);
    std::vector<std::vector<int64_t>> rl2;
    ReadBin_int(datapath + "03_rl2/tbnet_rl2_bs1_" +
                std::to_string(index) + ".bin", rl2, DATA_SIZE_39);
    std::vector<std::vector<int64_t>> his;
    ReadBin_int(datapath + "04_his/tbnet_his_bs1_" +
                std::to_string(index) + ".bin", his, DATA_SIZE_39);
    std::vector<std::vector<float>> rate;
    ReadBin_float(datapath + "05_rate/tbnet_rate_bs1_" +
                  std::to_string(index) + ".bin", rate, DATA_SIZE_1);

    if (ret != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    MxBase::TensorBase tensorBase0;
    APP_ERROR ret1 = VectorToTensorBase_int(item, tensorBase0, SHAPE[0]);
    inputs.push_back(tensorBase0);
    MxBase::TensorBase tensorBase1;
    VectorToTensorBase_int(rl1, tensorBase1, SHAPE[1]);
    inputs.push_back(tensorBase1);
    MxBase::TensorBase tensorBase2;
    VectorToTensorBase_int(ety, tensorBase2, SHAPE[2]);
    inputs.push_back(tensorBase2);
    MxBase::TensorBase tensorBase3;
    VectorToTensorBase_int(rl2, tensorBase3, SHAPE[3]);
    inputs.push_back(tensorBase3);
    MxBase::TensorBase tensorBase4;
    VectorToTensorBase_int(his, tensorBase4, SHAPE[4]);
    inputs.push_back(tensorBase4);
    MxBase::TensorBase tensorBase5;
    VectorToTensorBase_float(rate, tensorBase5, SHAPE[5]);
    inputs.push_back(tensorBase5);

    if (ret1 != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret1 << ".";
        return ret1;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret3 = Inference(inputs, outputs_tb);

    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret3 != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret3 << ".";
        return ret3;
    }
    for (size_t i = 0; i < outputs_tb.size(); ++i) {
        if (!outputs_tb[i].IsHost()) {
            outputs_tb[i].ToHost();
        }
    }
    WriteResult(std::to_string(index), outputs_tb);
}

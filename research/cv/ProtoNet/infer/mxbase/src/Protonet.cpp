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
#include "Protonet.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

const uint32_t EACH_LABEL_LENGTH = 4;
const uint32_t MAX_LENGTH = 313600;


APP_ERROR Protonet::Init(const InitParam &initParam) {
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


APP_ERROR Protonet::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR Protonet::ReadTensorFromFile(const std::string &file, uint32_t *data, uint32_t size) {
    if (data == NULL || size < MAX_LENGTH) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    std::ifstream infile;
    // open data file
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    // check data file validity
    if (infile.fail()) {
        LogError << "Failed to open data file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char*>(data), sizeof(uint32_t) * MAX_LENGTH);
    infile.close();
    return APP_ERR_OK;
}


APP_ERROR Protonet::ReadInputTensor(const std::string &fileName, std::vector<MxBase::TensorBase> *inputs) {
    uint32_t data[MAX_LENGTH] = {0};
    APP_ERROR ret = ReadTensorFromFile(fileName, data, MAX_LENGTH);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }

    const uint32_t dataSize = modelDesc_.inputTensors[0].tensorSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, MAX_LENGTH};
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT32));
    return APP_ERR_OK;
}


APP_ERROR Protonet::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);

    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR  Protonet::WriteResult(const std::string& imageFile, std::vector<MxBase::TensorBase> &outputs) {
    std::string infer_result_path;
    infer_result_path = "./result/";
    for (size_t i = 0; i < outputs.size(); ++i) {
        APP_ERROR ret = outputs[i].ToHost();
        if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
        }
        void *netOutput = outputs[i].GetBuffer();
        std::vector<uint32_t> out_shape = outputs[i].GetShape();
        std::string outFileName = infer_result_path + "/" + imageFile;
        FILE *outputFile = fopen(outFileName.c_str(), "wb");
        fwrite(netOutput, out_shape[0]*out_shape[1], sizeof(float), outputFile);
        fclose(outputFile);
        outputFile = nullptr;
    }
    return APP_ERR_OK;
}


APP_ERROR Protonet::Process(const std::string &inferPath, const std::string &fileName) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputIdsFile = inferPath + fileName;
    APP_ERROR ret = ReadInputTensor(inputIdsFile, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input ids failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> outputs = {};
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    ret = WriteResult(fileName, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Write result failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

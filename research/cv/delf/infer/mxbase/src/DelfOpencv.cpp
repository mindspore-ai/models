/*
 * Copyright 2022. Huawei Technologies Co., Ltd.
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

#include "DelfOpencv.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include <opencv2/dnn.hpp>
#include "MxBase/Log/Log.h"
const uint32_t EACH_LABEL_LENGTH = 4;
const uint32_t MAX_LENGTH = 307200;
void PrintTensorShape(const std::vector<MxBase::TensorDesc> &tensorDescVec, const std::string &tensorName) {
    LogInfo << "The shape of " << tensorName << " is as follows:";
    for (size_t i = 0; i < tensorDescVec.size(); ++i) {
        LogInfo << "  Tensor " << i << ":";
        for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
            LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
        }
    }
}

APP_ERROR DelfOpencv::Init(const InitParam& initParam) {
    this->deviceId_ = initParam.deviceId;
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
    this->model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = this->model_->Init(initParam.modelPath, this->modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
    PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");

    return APP_ERR_OK;
}

APP_ERROR DelfOpencv::DeInit() {
    this->model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR DelfOpencv::ReadTensorFromFile(const std::string &file, uint32_t *data, uint32_t size) {
    if (data == NULL) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    std::ifstream infile;
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    // check data file validity
    if (infile.fail()) {
        LogError << "Failed to open data file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.seekg(0, infile.end);
    int length = infile.tellg();
    infile.seekg(0, infile.beg);
    infile.read(reinterpret_cast<char*>(data), length);
    LogInfo <<"ssswww";
    infile.close();

    return APP_ERR_OK;
}

APP_ERROR DelfOpencv::ReadTensorFromFile1(const std::string& file, float* data) {
    if (data == NULL) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    std::ifstream infile;
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    // check data file validity
    if (infile.fail()) {
        LogError << "Failed to open data file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char*>(data), sizeof(float) * this->inputDataSize_);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR DelfOpencv::ReadInputTensor(const std::string& fileName,
                                            std::vector<MxBase::TensorBase> *inputs) {
    float *data = new float[this->inputDataSize_];
    APP_ERROR ret = ReadTensorFromFile1(fileName, data);
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
    std::vector<uint32_t> shape = {7, 3, 2048, 2048};
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32));
    return APP_ERR_OK;
}

APP_ERROR DelfOpencv::Inference(const std::vector<MxBase::TensorBase>& inputs,
    std::vector<MxBase::TensorBase>* outputs) {
    auto dtypes = this->model_->GetOutputDataType();
    for (size_t i = 0; i < this->modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)this->modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i],
            MxBase::MemoryData::MemoryType::MEMORY_DEVICE, this->deviceId_);
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
    // save time
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;

    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR  DelfOpencv::WriteResult(const std::string& imageFile, std::vector<MxBase::TensorBase> &outputs) {
    std::string infer_result_path;
    infer_result_path = "./results/result_Files";
    for (size_t i = 0; i < outputs.size(); ++i) {
        APP_ERROR ret = outputs[i].ToHost();
        if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
        }
        void *netOutput = outputs[i].GetBuffer();
        std::vector<uint32_t> out_shape = outputs[i].GetShape();
        std::string fileName(imageFile);
        std::string outFileName = infer_result_path + "/" + std::to_string(i) + "_" + fileName;
        LogInfo << outFileName;
        std::cout << outFileName << std::endl;
        FILE *outputFile = fopen(outFileName.c_str(), "wb");
        fwrite(netOutput, out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3], sizeof(float), outputFile);
        fclose(outputFile);
        outputFile = nullptr;
    }
    return APP_ERR_OK;
}

APP_ERROR DelfOpencv::Process(const std::string& inferPath,
    const std::string& fileName) {
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

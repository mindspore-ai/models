/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd.
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

#include "RetinaFaceOpencv.h"
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR RetinaFaceOpencv::Init(const InitParam& initParam) {
    this->deviceId_ = initParam.deviceId;
    this->resultPath_ = initParam.resultPath;
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
    this->dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = this->dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }
    this->model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = this->model_->Init(initParam.modelPath, this->modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    uint32_t input_data_size = 1;
    for (size_t j = 0; j < this->modelDesc_.inputTensors[0].tensorDims.size(); ++j) {
        this->inputDataShape_[j] = (uint32_t)this->modelDesc_.inputTensors[0].tensorDims[j];
        input_data_size *= this->inputDataShape_[j];
    }

    this->inputDataSize_ = input_data_size;
    std::cout << "input_data_size:" << this->inputDataSize_ << std::endl;

    return APP_ERR_OK;
}

APP_ERROR RetinaFaceOpencv::DeInit() {
    this->dvppWrapper_->DeInit();
    this->model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR RetinaFaceOpencv::ReadTensorFromFile(const std::string& file, float* data) {
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

APP_ERROR RetinaFaceOpencv::ReadInputTensor(const std::string& fileName,
    std::vector<MxBase::TensorBase>* inputs) {
    // float data[this->inputDataSize_] = { 0 };
    float *data = new float[this->inputDataSize_];

    APP_ERROR ret = ReadTensorFromFile(fileName, data);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }
    const uint32_t dataSize = this->modelDesc_.inputTensors[0].tensorSize;

    // test(float:input_data_size * 4)
    std::cout << "dataSize:" << dataSize << std::endl;

    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, this->deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    // format of numpy data when preprocessing is also float32
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, this->inputDataShape_, MxBase::TENSOR_DTYPE_FLOAT32));
    return APP_ERR_OK;
}

APP_ERROR RetinaFaceOpencv::Inference(const std::vector<MxBase::TensorBase>& inputs,
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

APP_ERROR RetinaFaceOpencv::SaveResult(const std::string& result,
std::vector<MxBase::TensorBase> outputs) {
    // do not need landmarks
    for (size_t i = 0; i < outputs.size() - 1; ++i) {
        APP_ERROR ret = outputs[i].ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "to host fail.";
            return ret;
        }
        auto *netOutput = reinterpret_cast<float *>(outputs[i].GetBuffer());
        std::vector<uint32_t> out_shape = outputs[i].GetShape();

        // create bin and add
        FILE *outputFile_ = fopen(result.c_str(), "ab");
            fwrite(netOutput, out_shape[0] * out_shape[1] * out_shape[2], sizeof(float), outputFile_);
            fclose(outputFile_);
        }
    return APP_ERR_OK;
}

APP_ERROR RetinaFaceOpencv::Process(const std::string& inferPath,
    const std::string& fileName) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputBinFile = inferPath + fileName;
    APP_ERROR ret = ReadInputTensor(inputBinFile, &inputs);

    if (ret != APP_ERR_OK) {
        LogError << "Read inputbinfile failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> outputs = {};
    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    // save time
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::string finalRetPath = "./mxbase_out/" + fileName;
    ret = SaveResult(finalRetPath, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "SaveResult failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

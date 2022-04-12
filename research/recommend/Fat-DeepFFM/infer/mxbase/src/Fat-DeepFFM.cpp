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
#include "Fat-DeepFFM.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

const uint32_t Spare_LENGTH = 26000;
const uint32_t Dense_LENGTH = 13000;
const uint32_t Label_LENGTH = 1000;


APP_ERROR DeepFFMBase::Init(const InitParam &initParam) {
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
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
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

APP_ERROR DeepFFMBase::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR DeepFFMBase::ReadTensorFromFile(const std::string &file, float *data, uint32_t size) {
    if (data == NULL) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    std::ifstream infile;
    // open file
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    // check file validity
    if (infile.fail()) {
        LogError << "Failed to open label file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char*>(data), sizeof(float) * size);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR DeepFFMBase::ReadInputTensor(const std::string &fileName, uint32_t index,
                                        std::vector<MxBase::TensorBase> *inputs, uint32_t size,
                                        MxBase::TensorDataType type) {
    float* data = new float[size];
    APP_ERROR ret = ReadTensorFromFile(fileName, data, size);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }
    const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;
    LogInfo << "dataSize: " << dataSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1000, size/1000};
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, type));
    return APP_ERR_OK;
}

APP_ERROR DeepFFMBase::Inference(const std::vector<MxBase::TensorBase> &inputs,
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

APP_ERROR DeepFFMBase::PostProcess(std::vector<MxBase::TensorBase> *outputs, std::vector<float> *result) {
    MxBase::TensorBase &tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // check tensor is available
    auto outputShape = tensor.GetShape();
    uint32_t length = outputShape[0];
    uint32_t classNum = outputShape[1];
    LogInfo << "output shape is: " << outputShape[0] << " "<< outputShape[1] << std::endl;

    void* data = tensor.GetBuffer();
    for (uint32_t i = 0; i < length; i++) {
        for (uint32_t j = 0; j < classNum; j++) {
            float value = *(reinterpret_cast<float*>(data) + i * classNum + j);
            result->push_back(value);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR DeepFFMBase::getAucResult(const std::string &labelFile, std::vector<float> *result) {
    float* label = new float[Label_LENGTH];
    APP_ERROR ret = ReadTensorFromFile(labelFile, label, Label_LENGTH);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }
    LogInfo << "label: " << label[0];
    std::ofstream fp_l("label.txt", std::ofstream::app);
    std::ofstream fp_r("pred.txt", std::ofstream::app);
    if (fp_l.fail() || fp_r.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    for (uint32_t i = 0; i < Label_LENGTH; i++) {
        fp_l << label[i] << " ";
        fp_r << result->at(i) << " ";
    }
    fp_l << std::endl;
    fp_r << std::endl;
    fp_l.close();
    fp_r.close();
    return APP_ERR_OK;
}


APP_ERROR DeepFFMBase::Process(const std::string &inferPath, const std::string &fileName) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::string spareFile = inferPath + "batch_spare/" + fileName;
    LogInfo << "read file name: " << spareFile;
    APP_ERROR ret = ReadInputTensor(spareFile, 0, &inputs, Spare_LENGTH, MxBase::TENSOR_DTYPE_UINT32);
    if (ret != APP_ERR_OK) {
        LogError << "Read input spare failed, ret=" << ret << ".";
        return ret;
    }
    std::string denseFile = inferPath + "batch_dense/" + fileName;
    LogInfo << "read file name: " << denseFile;
    ret = ReadInputTensor(denseFile, 1, &inputs, Dense_LENGTH, MxBase::TENSOR_DTYPE_FLOAT32);
    if (ret != APP_ERR_OK) {
        LogError << "Read input dense file failed, ret=" << ret << ".";
        return ret;
    }
    std::string labelFile = inferPath + "batch_labels/" + fileName;
    LogInfo << "read file name: " << labelFile;
    ret = ReadInputTensor(labelFile, 2, &inputs, Label_LENGTH, MxBase::TENSOR_DTYPE_FLOAT32);
    if (ret != APP_ERR_OK) {
        LogError << "Read label file failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> outputs = {};
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<float> result;
    ret = PostProcess(&outputs, &result);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    ret = getAucResult(labelFile, &result);
    if (ret != APP_ERR_OK) {
        LogError << "CalcF1Score read label failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

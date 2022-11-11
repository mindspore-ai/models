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
#include "TernaryBERT.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

const uint32_t MAX_LENGTH = 4096;

APP_ERROR TernaryBERT::Init(const InitParam &initParam) {
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

APP_ERROR TernaryBERT::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR TernaryBERT::ReadTensorFromFile(const std::string &file, uint32_t *data, uint32_t size) {
    // read file into data
    std::ifstream infile;
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    if (infile.fail()) {
        LogError << "Failed to open label file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char*>(data), sizeof(uint32_t) * size);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR TernaryBERT::ReadInputTensor(const std::string &fileName, \
                                        uint32_t index, \
                                        std::vector<MxBase::TensorBase> *inputs) {
    // read file into inputs
    uint32_t data[MAX_LENGTH] = {0};
    APP_ERROR ret = ReadTensorFromFile(fileName, data, MAX_LENGTH);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }

    const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {32, 128};
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT32));
    return APP_ERR_OK;
}

APP_ERROR TernaryBERT::Inference(const std::vector<MxBase::TensorBase> &inputs, \
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

    // model infer
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    return APP_ERR_OK;
}

APP_ERROR TernaryBERT::WriteResult(std::vector<MxBase::TensorBase> *outputs, const std::string &fileName) {
    MxBase::TensorBase &tensor = outputs->at(13);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }

    std::string resultPathName = "mxbase_result";
    // create result directory when it does not exit
    if (access(resultPathName.c_str(), 0) != 0) {
        int mkdir_ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (mkdir_ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << mkdir_ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }

    auto outputShape = tensor.GetShape();
    uint32_t length = outputShape[0];
    uint32_t classNum = outputShape[1];
    LogInfo << "output shape is: " << outputShape[0] << " "<< outputShape[1] << std::endl;

    void* data = tensor.GetBuffer();
    std::string outFileName = "mxbase_result/" + fileName;
    FILE * outputFile = fopen(outFileName.c_str(), "wb");
    fwrite(data, sizeof(float), length * classNum, outputFile);
    fclose(outputFile);

    return APP_ERR_OK;
}

APP_ERROR TernaryBERT::Process(const std::string &inferPath, const std::string &fileName) {
    // read file into inputs
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputContent = inferPath + "/00_input_ids/" + fileName;
    APP_ERROR ret = ReadInputTensor(inputContent, INPUT_IDS, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input content failed, ret=" << ret << ".";
        return ret;
    }

    std::string inputSenLen = inferPath + "/01_token_type_id/" + fileName;
    ret = ReadInputTensor(inputSenLen, TOKEN_TYPE_ID, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input sen_len failed, ret=" << ret << ".";
        return ret;
    }

    std::string inputAspect = inferPath + "/02_input_mask/" + fileName;
    ret = ReadInputTensor(inputAspect, INPUT_MASK, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input aspect failed, ret=" << ret << ".";
        return ret;
    }

    // infer and put result into outputs
    std::vector<MxBase::TensorBase> outputs = {};
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    // write result
    ret = WriteResult(&outputs, fileName);
    if (ret != APP_ERR_OK) {
        LogError << "save result failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/*
 * Copyright 2022 Huawei Technologies Co., Ltd.
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

#include "Melgan.h"
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "acl/acl.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR MELGAN::Init(const InitParam &initParam) {
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

APP_ERROR MELGAN::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR MELGAN::VectorToTensorBase(const std::vector<std::vector<std::vector<float>>> &input_x,
                                                                 MxBase::TensorBase *tensorBase) {
    const uint32_t dataSize = modelDesc_.inputTensors[0].tensorSize / 4;
    float *metaFeatureData = new float[dataSize];
    uint32_t idx = 0;
    for (size_t bs = 0; bs < input_x.size(); bs++) {
        for (size_t c = 0; c < input_x[0].size(); c++) {
            for (size_t d = 0; d < input_x[0][0].size(); d++) {
                metaFeatureData[idx++] = input_x[bs][c][d];
            }
        }
    }
    MxBase::MemoryData memoryDataDst(dataSize * 4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(metaFeatureData),
                                     dataSize * 4, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, 80, 240};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR MELGAN::Inference(const std::vector<MxBase::TensorBase> &inputs,
                            std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); i++) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); j++) {
            shape.push_back((uint32_t) modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        (*outputs).push_back(tensor);
    }

    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MELGAN::SaveInferResult(std::vector<float> *outputs, std::vector<MxBase::TensorBase> *inputs) {
    MxBase::TensorBase &tensor = inputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }

    // check tensor is available
    auto outputShape = tensor.GetShape();
    uint32_t length = outputShape[2];
    LogInfo << "output shape is: (" << outputShape[0] << ',' << outputShape[1] << ',' << outputShape[2] << ')';
    void *data = tensor.GetBuffer();
    for (uint32_t i = 0; i < length; i++) {
        float value = *(reinterpret_cast<float *>(data) + i);
        outputs->emplace_back(value);
    }
    return APP_ERR_OK;
}


APP_ERROR MELGAN::Process(const std::string &fileName, const std::vector<std::vector<std::vector<float>>> &input_x,
                            InitParam &initParam, std::vector<float> output) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs;
    MxBase::TensorBase tensorBase;
    auto ret = VectorToTensorBase(input_x, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);
    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs);

    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    ret = SaveInferResult(&output, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Save model infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    ret = WriteResult(fileName, output);
    if (ret != APP_ERR_OK) {
        LogError << "WriteResult failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR MELGAN::WriteResult(const std::string &fileName,
                              const std::vector<float> &output) {
    std::string resultPathName = "output";
    // create result directory when it does not exit
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName
                     << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }
    // create result file under result directory
    resultPathName = resultPathName + "/restruction_" + fileName;
    std::ofstream tfile(resultPathName, std::ofstream::app);
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }

    for (uint32_t i = 0; i < output.size(); i++) {
        tfile << std::to_string(output[i]) << " ";
    }
    tfile.close();
    return APP_ERR_OK;
}

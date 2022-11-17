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

#include <unistd.h>
#include <sys/stat.h>
#include <memory>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include "Deepsort.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR DEEPSORT::Init(const InitParam &initParam) {
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

APP_ERROR DEEPSORT::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR DEEPSORT::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i],  MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
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
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR DEEPSORT::PostProcess(std::vector<float> &det, const std::vector<float> &feature) {
    std::copy(feature.begin(), feature.end(),
              std::back_inserter(det));
    return APP_ERR_OK;
}

APP_ERROR DEEPSORT::SaveInferResult(std::vector<float> &batchFeaturePaths,
 const std::vector<MxBase::TensorBase> &inputs) {
    LogInfo << "Infer results before postprocess:\n";
    for (auto retTensor : inputs) {
        LogInfo << "Tensor description:\n" << retTensor.GetDesc();
        std::vector<uint32_t> shape = retTensor.GetShape();
        uint32_t N = shape[0];
        uint32_t C = shape[1];
        if (!retTensor.IsHost()) {
            LogInfo << "this tensor is not in host. Now deploy it to host";
            retTensor.ToHost();
        }
        void* data = retTensor.GetBuffer();
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < C; j++) {
                float value = *(reinterpret_cast<float*>(data) + i * C + j);
                batchFeaturePaths.emplace_back(value);
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR DEEPSORT::WriteResult(const std::vector<float> &outputs, const std::string &result_path) {
    std::ofstream outfile(result_path, std::ios::app);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }

    std::string tmp;
    for (auto x : outputs) {
        tmp += std::to_string(x) + " ";
    }
    tmp = tmp.substr(0, tmp.size()-1);
    outfile << tmp << std::endl;
    outfile.close();
    return APP_ERR_OK;
}

APP_ERROR DEEPSORT::ReadInputTensor(const std::string &fileName, MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }
    float *metaFeatureData = new float[dataSize];
    std::ifstream infile;
    // open data file
    infile.open(fileName, std::ios_base::in | std::ios_base::binary);
    // check data file validity
    if (infile.fail()) {
        LogError << "Failed to open data file: " << fileName << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char*>(metaFeatureData), sizeof(float) * dataSize);
    infile.close();

    MxBase::MemoryData memoryDataDst(dataSize * 4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(metaFeatureData),
    dataSize * 4, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    auto ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, 3, 128, 64};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR DEEPSORT::Process(const std::string &imagePath, std::vector<float> &det, const std::string &result_path) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    std::vector<float> batchFeaturePaths;
    MxBase::TensorBase tensorBase;
    auto ret = ReadInputTensor(imagePath, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "Read input ids failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);
    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, outputs);

    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    ret = SaveInferResult(batchFeaturePaths, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Save model infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    ret = PostProcess(det, batchFeaturePaths);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    ret = WriteResult(det, result_path);
    if (ret != APP_ERR_OK) {
        LogError << "WriteResult failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

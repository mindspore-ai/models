/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd
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
#include <math.h>
#include <memory>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include "MMoE.h"
#include "half.hpp"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

using half_float::half;

APP_ERROR MMoE::Init(const InitParam &initParam) {
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

APP_ERROR MMoE::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

template<class dtype>
APP_ERROR MMoE::VectorToTensorBase(const std::vector<std::vector<dtype>> &input, uint32_t inputId
                                     , MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = modelDesc_.inputTensors[inputId].tensorDims[1];
    dtype *metaFeatureData = new dtype[dataSize];
    uint32_t idx = 0;
    for (size_t bs = 0; bs < input.size(); bs++) {
        for (size_t c = 0; c < input[bs].size(); c++) {
            metaFeatureData[idx++] = input[bs][c];
        }
    }

    MxBase::MemoryData memoryDataDst(dataSize * 2, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(metaFeatureData), dataSize * 2
                                     , MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, dataSize};
    if (typeid(dtype) == typeid(half)) {
        tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT16);
    } else {
        tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_INT32);
    }
    return APP_ERR_OK;
}

APP_ERROR MMoE::Inference(const std::vector<MxBase::TensorBase> &inputs,
                             std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); i++) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); j++) {
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

APP_ERROR MMoE::PostProcess(std::vector<std::vector<half>> &income_preds
                            , std::vector<std::vector<half>> &married_preds
                            , const std::vector<MxBase::TensorBase> &inputs) {
    size_t index = 0;
    for (auto retTensor : inputs) {
        std::vector<uint32_t> shape = retTensor.GetShape();
        uint32_t N = shape[0];
        uint32_t C = shape[1];
        if (!retTensor.IsHost()) {
            retTensor.ToHost();
        }
        void* data = retTensor.GetBuffer();
        std::vector<half> temp;
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < C; j++) {
                half value = *(reinterpret_cast<half*>(data) + i * C + j);
                temp.push_back(value);
            }
        }
        if (index == 0) {
            income_preds.emplace_back(temp);
        } else {
            married_preds.emplace_back(temp);
        }
        index++;
    }
    return APP_ERR_OK;
}

APP_ERROR MMoE::PrintInputInfo(std::vector<MxBase::TensorBase> inputs) {
    LogInfo << "input size: " << inputs.size();
    for (size_t i = 0; i < inputs.size(); i++) {
        // check tensor is available
        MxBase::TensorBase &tensor_input = inputs[i];
        auto inputShape = tensor_input.GetShape();
        uint32_t inputDataType = tensor_input.GetDataType();
        LogInfo << "input_" + std::to_string(i) + "_shape is: " << inputShape[0]
                << " " << inputShape[1] << " " << inputShape.size();
        LogInfo << "input_" + std::to_string(i) + "_dtype is: " << inputDataType;
    }
    return APP_ERR_OK;
}

APP_ERROR MMoE::Process(const std::vector<std::vector<half>> &data, const InitParam &initParam
                        , std::vector<std::vector<half>> &income_preds
                        , std::vector<std::vector<half>> &married_preds) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    APP_ERROR ret;
    MxBase::TensorBase tensorBase;
    ret = VectorToTensorBase(data, 0, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);

    // run inference
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    ret = PostProcess(income_preds, married_preds, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Save model infer results into file failed. ret = " << ret << ".";
        return ret;
    }
}

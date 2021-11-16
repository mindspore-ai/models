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

#include "MxNCFBase.h"
#include <fstream>
#include <cmath>
#include <numeric>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/Log/Log.h"

const uint32_t MAX_LENGTH = 160000;

APP_ERROR NCFBase::Init(const InitParam &initParam) {
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

APP_ERROR NCFBase::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR NCFBase::ReadTensorFromFile(const std::string &file, uint32_t *data, uint32_t size) {
    if (data == NULL || size < MAX_LENGTH) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    std::ifstream infile;
    // open file
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    // check file validity
    if (infile.fail()) {
        LogError << "Failed to open file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char *>(data), sizeof(uint32_t) * MAX_LENGTH);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR NCFBase::ReadInputTensor(const std::string &fileName, enum DataIndex di,
                                   std::vector<MxBase::TensorBase> *inputs) {
    uint32_t data[MAX_LENGTH] = {0};
    uint32_t index = di;
    APP_ERROR ret = ReadTensorFromFile(fileName, data, MAX_LENGTH);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }

    const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, MAX_LENGTH};
    if (di == INPUT_USERS || di == INPUT_ITEMS) {
        inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_INT32));
    }
    if (di == INPUT_MASKS) {
        inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32));
    }
    return APP_ERR_OK;
}

APP_ERROR NCFBase::Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs) {
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

APP_ERROR NCFBase::PostProcess(std::vector<MxBase::TensorBase> *outputs, std::vector<int32_t> *hitRates,
                               std::vector<double> *ndcgRates) {
    MxBase::TensorBase &tensor_indices = outputs->at(0);
    APP_ERROR ret = tensor_indices.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor_0 deploy to host failed.";
        return ret;
    }
    // check tensor is available
    auto outputShape = tensor_indices.GetShape();
    uint32_t userNum = outputShape[0];
    uint32_t topK = outputShape[1];
    LogInfo << "output_0 shape is: " << outputShape[0] << " " << outputShape[1] << std::endl;
    void *indices = tensor_indices.GetBuffer();

    MxBase::TensorBase &tensor_items = outputs->at(1);
    ret = tensor_items.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor_1 deploy to host failed.";
        return ret;
    }
    // check tensor is available
    outputShape = tensor_items.GetShape();
    uint32_t itemNum = outputShape[1];
    LogInfo << "output_1 shape is: " << outputShape[0] << " " << outputShape[1] << std::endl;
    void *items = tensor_items.GetBuffer();

    MxBase::TensorBase &tensor_weights = outputs->at(2);
    ret = tensor_weights.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor_2 deploy to host failed.";
        return ret;
    }
    // check tensor is available
    outputShape = tensor_weights.GetShape();
    LogInfo << "output_2 shape is: " << outputShape[0] << std::endl;
    void *weights = tensor_weights.GetBuffer();

    int32_t hit;
    double ndcg;
    for (uint32_t i = 0; i < userNum; i++) {
        bool isValid = *(reinterpret_cast<bool *>(weights) + i);
        if (isValid == true) {
            std::vector<int32_t> recommends = {};
            int32_t gt_item = *(reinterpret_cast<int32_t *>(items) + (i + 1) * itemNum - 1);
            for (uint32_t j = 0; j < topK; j++) {
                int32_t index = *(reinterpret_cast<int32_t *>(indices) + i * topK + j);
                int32_t value = *(reinterpret_cast<int32_t *>(items) + i * itemNum + index);
                recommends.push_back(value);
            }
            hit = 0;
            ndcg = 0.0;
            for (uint32_t rank = 0; rank < recommends.size(); rank++) {
                if (gt_item == recommends[rank]) {
                    hit = 1;
                    ndcg = 1.0 / (log(rank + 2.0) / log(2.0));
                    break;
                }
            }
            hitRates->push_back(hit);
            ndcgRates->push_back(ndcg);
        }
    }

    return APP_ERR_OK;
}

APP_ERROR NCFBase::Process(const std::string &inferPath, const std::string &fileName, bool eval) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputUsersFile = inferPath + "tensor_0/" + fileName;
    APP_ERROR ret = ReadInputTensor(inputUsersFile, INPUT_USERS, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input users failed, ret=" << ret << ".";
        return ret;
    }
    std::string inputItemsFile = inferPath + "tensor_1/" + fileName;
    ret = ReadInputTensor(inputItemsFile, INPUT_ITEMS, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input items file failed, ret=" << ret << ".";
        return ret;
    }
    std::string inputMasksFile = inferPath + "tensor_2/" + fileName;
    ret = ReadInputTensor(inputMasksFile, INPUT_MASKS, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input masks file failed, ret=" << ret << ".";
        return ret;
    }

    LogInfo << "input size: " << inputs.size();
    MxBase::TensorBase &tensor_input_0 = inputs[0];
    // check tensor is available
    auto inputShape = tensor_input_0.GetShape();
    uint32_t inputDataType = tensor_input_0.GetDataType();
    LogInfo << "input_0 shape is: " << inputShape[0] << " " << inputShape[1] << " " << inputShape.size();
    LogInfo << "input_0 dtype is: " << inputDataType;

    MxBase::TensorBase &tensor_input_1 = inputs[1];
    // check tensor is available
    inputShape = tensor_input_1.GetShape();
    inputDataType = tensor_input_1.GetDataType();
    LogInfo << "input_1 shape is: " << inputShape[0] << " " << inputShape[1] << " " << inputShape.size();
    LogInfo << "input_1 dtype is: " << inputDataType;

    MxBase::TensorBase &tensor_input_2 = inputs[2];
    // check tensor is available
    inputShape = tensor_input_2.GetShape();
    inputDataType = tensor_input_2.GetDataType();
    LogInfo << "input_2 shape is: " << inputShape[0] << " " << inputShape[1] << " " << inputShape.size();
    LogInfo << "input_2 dtype is: " << inputDataType;

    std::vector<MxBase::TensorBase> outputs = {};
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<int32_t> hitRates;
    std::vector<double> ndcgRates;
    ret = PostProcess(&outputs, &hitRates, &ndcgRates);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    if (eval) {
        double hr_sum = std::accumulate(hitRates.begin(), hitRates.end(), 0.0);
        double ndcgr_sum = std::accumulate(ndcgRates.begin(), ndcgRates.end(), 0.0);
        hr_num += hitRates.size();
        ndcgr_num += ndcgRates.size();
        hr_total += hr_sum;
        ndcgr_total += ndcgr_sum;
        double hr = hr_sum / hitRates.size();
        double ndcgr = ndcgr_sum / ndcgRates.size();
        LogInfo << "HR = " << hr << ", NDCG = " << ndcgr;
    }

    return APP_ERR_OK;
}

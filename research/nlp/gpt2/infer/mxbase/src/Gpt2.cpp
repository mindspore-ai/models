/*
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
#include "Gpt2.h"

#include <unistd.h>
#include <sys/stat.h>

#include <iostream>
#include <iomanip>
#include <map>
#include <fstream>

#include "acl/acl.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

const uint32_t SEQ_LENGTH = 1024;
const uint32_t VOCAB_SIZE = 50257;

namespace {
    uint16_t UINT32_SIZE = 4;
}

APP_ERROR gpt2::VectorToTensorBase_int32(const std::vector<uint32_t> &batchFeatureVector,
                                    MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 1;
    std::vector<uint32_t> shape = {};
    shape.push_back(1);
    shape.push_back(batchFeatureVector.size());

    for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    uint32_t *metaFeatureData = new uint32_t[dataSize];
    uint32_t idx = 0;

    for (size_t w = 0; w < batchFeatureVector.size(); w++) {
      metaFeatureData[idx++] = batchFeatureVector[w];
    }
    MxBase::MemoryData memoryDataDst(dataSize * UINT32_SIZE, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(metaFeatureData, dataSize * UINT32_SIZE, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_INT32);
    return APP_ERR_OK;
}

APP_ERROR gpt2::Init(const InitParam &initParam) {
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

    model_gpt2 = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_gpt2->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR gpt2::DeInit() {
    model_gpt2->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR gpt2::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                 std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_gpt2->GetOutputDataType();
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
    APP_ERROR ret = model_gpt2->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR gpt2::PostProcess(std::vector<MxBase::TensorBase> *outputs,
                            std::vector<MxBase::TensorBase> *inputs,
                            std::vector<double> *loss) {
    MxBase::TensorBase &tensor = outputs->at(0);
    MxBase::TensorBase &label_ids = inputs->at(0);
    MxBase::TensorBase &input_mask = inputs->at(1);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    APP_ERROR ret1 = label_ids.ToHost();
    if (ret1 != APP_ERR_OK) {
        LogError << GetError(ret1) << "label_ids deploy to host failed.";
        return ret1;
    }
    APP_ERROR ret2 = input_mask.ToHost();
    if (ret2 != APP_ERR_OK) {
        LogError << GetError(ret2) << "input_mask deploy to host failed.";
        return ret2;
    }
    // check tensor is available
    auto outputShape = tensor.GetShape();
    uint32_t batch_size = outputShape[0];
    uint32_t seq_length = outputShape[1];
    uint32_t vocab_size = outputShape[2];
    LogInfo << "shape of the infer result: " << batch_size << " "<<
            seq_length << " " << vocab_size << std::endl;

    std::vector<double> per_example_loss(SEQ_LENGTH - 1, 0);
    void* data_inputs = label_ids.GetBuffer();
    void* data_mask = input_mask.GetBuffer();
    void* data_outputs = tensor.GetBuffer();
    double valid_loss_sum = 0;
    double valid_element_sum = 0;
    double loss_example = 0;

    for (uint32_t i = 0; i < SEQ_LENGTH - 1; i++) {
    per_example_loss[i] = - *(reinterpret_cast<float*>(data_outputs) +
                            *(reinterpret_cast<uint32_t*>(data_inputs) + i + 1) + i * VOCAB_SIZE);
    valid_loss_sum += per_example_loss[i] * *(reinterpret_cast<uint32_t*>(data_mask) + i + 1);
    valid_element_sum += *(reinterpret_cast<uint32_t*>(data_mask) + i + 1);
    }
    loss_example = valid_loss_sum / valid_element_sum;

    loss->push_back(loss_example);

    return APP_ERR_OK;
}

APP_ERROR gpt2::WriteResult(const std::string &fileName, const std::vector<double> &loss) {
    // create result directory when it resultPathNamedoes not exit
    if (access(fileName.c_str(), 0) != 0) {
        int ret = mkdir(fileName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << fileName << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }
    // create result file under result directory
    std::string resultPathName = fileName + "result.txt";
    std::ofstream tfile(resultPathName, std::ofstream::app);
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::string secorePathName = fileName + "score.txt";
    std::ofstream tfile_score(secorePathName, std::ofstream::app);
    if (tfile_score.fail()) {
        LogError << "Failed to open result file: " << secorePathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    double PPL = 0;
    double num = 0;
    // write inference result into file
    LogInfo << "==============================================================";
    LogInfo << "infer result of " << fileName << " is: ";
    for (auto &item : loss) {
    PPL += item;
    num += 1;
    tfile  << item << std::endl;
    tfile_score  << exp(item) << std::endl;
    }
    PPL /= num;
    LogInfo << " | Current Loss : " << PPL <<".";
    LogInfo << " | Current PPL : " << exp(PPL) << ".";
    LogInfo << "==============================================================";
    tfile.close();
    tfile_score.close();
    return APP_ERR_OK;
}

APP_ERROR gpt2::Process(const std::string &inferPath,
                        const std::vector<uint32_t> &input_ids,
                        const std::vector<uint32_t> &input_mask,
                        const std::vector<uint32_t> &label_ids,
                        const InitParam &initParam,
                        float outputs) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs_tb = {};

    MxBase::TensorBase tensorBase1;
    MxBase::TensorBase tensorBase2;
    MxBase::TensorBase tensorBase3;

    APP_ERROR ret = VectorToTensorBase_int32(input_ids, tensorBase1);
    if (ret != APP_ERR_OK) {
        LogError << "input_ids ToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    APP_ERROR ret1 = VectorToTensorBase_int32(input_mask, tensorBase2);
    if (ret1 != APP_ERR_OK) {
        LogError << "input_mask ToTensorBase failed, ret=" << ret1 << ".";
        return ret1;
    }

    APP_ERROR ret2 = VectorToTensorBase_int32(label_ids, tensorBase3);
    if (ret2 != APP_ERR_OK) {
        LogError << "label_ids ToTensorBase failed, ret=" << ret2 << ".";
        return ret2;
    }

    inputs.push_back(tensorBase1);
    inputs.push_back(tensorBase2);
    inputs.push_back(tensorBase3);

    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs_tb);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    } else {
        LogInfo << "shape of the output:" << outputs_tb.size() << ".";
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;

    if (!outputs_tb[0].IsHost()) {
        outputs_tb[0].ToHost();
    }

    std::vector<double> loss;
    ret = PostProcess(&outputs_tb, &inputs, &loss);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    std::string resultPath = inferPath + "results/";
    ret = WriteResult(resultPath, loss);
    if (ret != APP_ERR_OK) {
        LogError << "save result failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}


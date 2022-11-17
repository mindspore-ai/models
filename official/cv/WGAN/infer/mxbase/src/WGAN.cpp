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
#include <random>
#include <ctime>
#include <cstddef>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "WGAN.h"

void PrintTensorShape(const std::vector<MxBase::TensorDesc> &tensorDescVec, const std::string &tensorName) {
    LogInfo << "The shape of " << tensorName << " is as follows:";
    for (size_t i = 0; i < tensorDescVec.size(); ++i) {
        LogInfo << "  Tensor " << i << ":";

        for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
            LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
        }
    }
}

APP_ERROR WGAN::Generate_input_Tensor(const model_info &modelInfo, std::vector<MxBase::TensorBase> *input) {
    // random generate input noise(model input data). dtype: float32
    static std::default_random_engine random_generate(time(nullptr));
    static std::normal_distribution<float> random_normal(0.0, 1.0);
    float *input_data = new float[modelInfo.noise_length];
    for (size_t i = 0; i < modelInfo.noise_length; ++i)
        input_data[i] = random_normal(random_generate);

    const uint32_t dataSize = modelDesc_.inputTensors[0].tensorSize;

    if (sizeof(*input_data) * modelInfo.noise_length != dataSize) {
        LogError << "Input data is invalid.";
        return 0;
    }
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(input_data), dataSize,
    MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, modelInfo.noise_length, 1, 1};
    input->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32));
    return APP_ERR_OK;
}

APP_ERROR WGAN::Init(const InitParam &initParam) {
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
    PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
    PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");
    return APP_ERR_OK;
}

APP_ERROR WGAN::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR WGAN::Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs) {
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
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR WGAN::PostProcess(const std::vector<MxBase::TensorBase> &infer_outputs,
    std::vector<float> *processed_result) {
    // Move tensor to host
    MxBase::TensorBase tensor = infer_outputs.at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }

    // process result
    void *data = tensor.GetBuffer();
    size_t data_size = tensor.GetSize();
    for (size_t i = 0; i < data_size; ++i) {
        float value = *(reinterpret_cast<float *>(data) + i);
        processed_result->push_back(value);
    }
    for (size_t i = 0; i < processed_result->size(); ++i)
        processed_result->at(i) = processed_result->at(i) * 0.5 * 255 + 0.5 * 255;

    std::cout << "infer result size :" << processed_result->size() << std::endl;

    return APP_ERR_OK;
}

APP_ERROR WGAN::WriteResult(const std::string &fileName, std::vector<float> *output_img_data) {
    // write result to a .txt file.
    const char *path = "../data/MxBase_result/result.txt";
    std::ofstream fout;
    std::cout << "file path" << path << std::endl;
    fout.open(path, std::ios::out);

    for (size_t i = 0; i < output_img_data->size(); ++i) {
        fout << output_img_data->at(i) << std::endl;
    }
    fout.close();

    return APP_ERR_OK;
}

APP_ERROR WGAN::Process(const model_info &modelInfo, const std::string &resultPath) {
    std::vector<MxBase::TensorBase> inputs, outputs = {};
    // preprocess
    APP_ERROR ret = Generate_input_Tensor(modelInfo, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Generate input ids failed, ret=" << ret << ".";
        return ret;
    }
    // inference
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "Inference success, ret=" << ret << ".";

    std::vector<float> processed_result;

    // post process
    ret = PostProcess(outputs, &processed_result);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    // Save infenrece result pictures to file.
    WriteResult(resultPath, &processed_result);

    return APP_ERR_OK;
}

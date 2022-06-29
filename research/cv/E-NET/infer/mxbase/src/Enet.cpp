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
#include "Enet.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include <vector>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"

std::vector<cv::Vec3b> cityspallete {
    {128, 64, 128},
    {244, 35, 232},
    {70, 70, 70},
    {102, 102, 156},
    {190, 153, 153},
    {153, 153, 153},
    {0, 130, 180},
    {220, 220, 0},
    {107, 142, 35},
    {152, 251, 152},
    {250, 170, 30},
    {220, 20, 60},
    {0, 0, 230},
    {119, 11, 32},
    {0, 0, 70},
    {0, 60, 100},
    {0, 80, 100},
    {255, 0, 0},
    {0, 0, 142},
    {0, 0, 0}};

APP_ERROR Enet::Init(const InitParam &initParam) {
    this->deviceId_ = initParam.deviceId;
    this->outputDataPath_ = initParam.outputDataPath;
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
    uint32_t input_data_size = 1;
    for (size_t j = 0; j < this->modelDesc_.inputTensors[0].tensorDims.size(); ++j) {
        this->inputDataShape_[j] = (uint32_t)this->modelDesc_.inputTensors[0].tensorDims[j];
        input_data_size *= this->inputDataShape_[j];
    }
    this->inputDataSize_ = input_data_size;

    return APP_ERR_OK;
}

APP_ERROR Enet::DeInit() {
    this->model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Enet::ReadTensorFromFile(const std::string &file, float *data) {
    if (data == NULL) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    std::ifstream infile;
    // open data file
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    // check data file validity
    if (infile.fail()) {
        LogError << "Failed to open data file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char *>(data), sizeof(float) * this->inputDataSize_);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR Enet::ReadInputTensor(const std::string &fileName, std::vector<MxBase::TensorBase> *inputs) {
    float data[this->inputDataSize_] = {0};
    APP_ERROR ret = ReadTensorFromFile(fileName, data);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }

    const uint32_t dataSize = modelDesc_.inputTensors[0].tensorSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, this->deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, this->inputDataShape_, MxBase::TENSOR_DTYPE_FLOAT32));
    return APP_ERR_OK;
}

APP_ERROR Enet::Inference(const std::vector<MxBase::TensorBase> &inputs,
                          std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = this->model_->GetOutputDataType();
    for (size_t i = 0; i < this->modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)this->modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, this->deviceId_);
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
    APP_ERROR ret = this->model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);

    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Enet::PostProcess(std::vector<MxBase::TensorBase> &inputs, cv::Mat &output) {
    MxBase::TensorBase &tensor = inputs[0];

    int channel = tensor.GetShape()[MxBase::VECTOR_SECOND_INDEX];
    int outputModelHeight = tensor.GetShape()[MxBase::VECTOR_THIRD_INDEX];
    int outputModelWidth = tensor.GetShape()[MxBase::VECTOR_FOURTH_INDEX];

    // argmax
    for (int h = 0; h < outputModelHeight; h++) {
        for (int w = 0; w < outputModelWidth; w++) {
            float max;
            int index = 0;
            std::vector<int> index_ori = {0, index, h, w};
            tensor.GetValue(max, index_ori);
            for (int c = 1; c < channel; c++) {
                float num_c;
                std::vector<int> index_cur = {0, c, h, w};
                tensor.GetValue(num_c, index_cur);
                if (num_c > max) {
                    index = c;
                    max = num_c;
                }
            }
            output.at<cv::Vec3b>(h, w) = cityspallete[index];
        }
    }
    return APP_ERR_OK;
}

APP_ERROR Enet::Process(const std::string &inferPath, const std::string &fileName) {
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

    ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << "ToHost failed, ret=" << ret << ".";
        return ret;
    }

    int outputModelHeight = outputs[0].GetShape()[MxBase::VECTOR_THIRD_INDEX];
    int outputModelWidth = outputs[0].GetShape()[MxBase::VECTOR_FOURTH_INDEX];
    cv::Mat output(outputModelHeight, outputModelWidth, CV_8UC3);
    ret = PostProcess(outputs, output);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    std::string outFileName = this->outputDataPath_ + "/" + fileName;
    size_t pos = outFileName.find_last_of(".");
    outFileName.replace(outFileName.begin() + pos, outFileName.end(), "_infer.png");
    cv::imwrite(outFileName, output);
    return APP_ERR_OK;
}

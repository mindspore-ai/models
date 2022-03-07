/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd.
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

#include "TALLInfer.h"
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <map>
#include <fstream>
#include <memory>
#include <vector>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "opencv2/opencv.hpp"

TALLInfer::TALLInfer(const uint32_t &deviceId, const std::string &modelPath) : deviceId_(deviceId) {
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        exit(-1);
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(deviceId_);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        exit(-1);
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        exit(-1);
    }
}

TALLInfer::~TALLInfer() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
}

// 模型推理
APP_ERROR TALLInfer::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                     std::vector<MxBase::TensorBase> &outputs) {
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
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR WriteResult(const std::string &imgPath,
                      std::vector<MxBase::TensorBase> *outputs, const std::string &output_path) {
    MxBase::TensorBase &tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // check tensor is available
    auto outputShape = tensor.GetShape();
    LogInfo << "output shape is: " << outputShape[0] << " "<< outputShape[1] << " "<< outputShape[2] << std::endl;

    void* data = tensor.GetBuffer();
    for (uint32_t i = 0; i < outputShape[0]; i++) {
        for (uint32_t j = 0; j < outputShape[1]; j++) {
            for (uint32_t k = 0; k < outputShape[1]; k++) {
                float value = *(reinterpret_cast<float*>(data) + i * outputShape[0] + j*outputShape[1] + k);
            }
        }
    }
    std::size_t found = imgPath.find_last_of("/\\");
    std::string outFileName = imgPath.substr(found+1);
    outFileName = outFileName.replace(outFileName.find(".data"), 5, ".bin");
    std::string outFilePath = output_path + "/" + outFileName;
    FILE * outputFile = fopen(outFilePath.c_str(), "wb");
    fwrite(data, 128*128*3, sizeof(float), outputFile);
    fclose(outputFile);
    return APP_ERR_OK;
}

APP_ERROR TALLInfer::ReadTensorFromFile(const std::string &file, float *data, uint32_t size) {
    // read file into data
    std::ifstream infile;
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    if (infile.fail()) {
        LogError << "Failed to open label file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char*>(data), sizeof(float) * size);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR TALLInfer::Process(const std::string &imgPath, const std::string &output_path) {
    const std::string img_name(imgPath.begin() + imgPath.find_last_of('/'), imgPath.end());
    uint32_t MAX_LENGTH = 128*17088;
    // read file into inputs
    float *data = new float[MAX_LENGTH];
    APP_ERROR ret = ReadTensorFromFile(imgPath, data, MAX_LENGTH);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    const uint32_t dataSize = MAX_LENGTH * 4;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {128, 17088};
    inputs.push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32));
    delete []data;

    std::vector<MxBase::TensorBase> outputs = {};

    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    WriteResult(imgPath, &outputs, output_path);
    return APP_ERR_OK;
}

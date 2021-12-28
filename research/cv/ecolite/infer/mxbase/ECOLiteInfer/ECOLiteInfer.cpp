/*
* Copyright(C) 2021. Huawei Technologies Co.,Ltd
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
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <map>
#include <fstream>
#include <memory>
#include <vector>
#include "opencv2/opencv.hpp"
#include "ECOLiteInfer.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

ECOLiteInfer::ECOLiteInfer(const uint32_t &deviceId, const std::string &modelPath) : deviceId_(deviceId) {
    LogInfo << "ECOLiteInfer Construct!!!";
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
    LogInfo << "ECOLiteInfer Construct End!!!";
}

ECOLiteInfer::~ECOLiteInfer() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
}

// 模型推理
APP_ERROR ECOLiteInfer::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            LogInfo << "shape:" << shape[j];
        }
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    LogInfo << "Before model_->ModelInference!!!!";
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR WriteResult(const std::string &imp, std::vector<MxBase::TensorBase> *oup, const std::string &output_path) {
    MxBase::TensorBase &tensor = oup->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // check tensor is available
    auto outputShape = tensor.GetShape();
    LogInfo << "output shape is: " << outputShape[0] << " "<< outputShape[1] << std::endl;

    void* data = tensor.GetBuffer();

    std::size_t found = imp.find_last_of("/\\");

    std::string outFileName = imp.substr(found+1);
    outFileName = outFileName.replace(outFileName.find("data"), 4, "predict");
    std::string outFilePath = output_path + "/" + outFileName;
    LogError << "outFilePath: " << outFilePath;

    FILE * outputFile = fopen(outFilePath.c_str(), "wb");
    fwrite(data, 16*101, sizeof(float), outputFile);
    fclose(outputFile);
    return APP_ERR_OK;
}

APP_ERROR ECOLiteInfer::ReadTensorFromFile(const std::string &file, float *data, uint32_t size) {
    // read file into data
    std::ifstream infile;
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    if (infile.fail()) {
        LogError << "Failed to open label file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    LogError << "Success to open label file: " << file << ".";
    infile.read(reinterpret_cast<char*>(data), sizeof(float) * size);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR ECOLiteInfer::Process(const std::string &imgPath, const std::string &output_path) {
    LogInfo << "READ File!!!!";
    uint32_t MAX_LENGTH = 16*12*224*224;
    // read file into inputs
    float *data = new float[MAX_LENGTH];
    APP_ERROR ret = ReadTensorFromFile(imgPath, data, MAX_LENGTH);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }
    LogError << "DATA: " << data << ".";
    std::vector<MxBase::TensorBase> inputs = {};
    const uint32_t dataSize = MAX_LENGTH * 4;
    LogError << "DataSize: " << dataSize << ".";
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {16, 12, 224, 224};
    inputs.push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32));
    delete []data;

    std::vector<MxBase::TensorBase> outputs = {};
    LogInfo << "Before Inference!!!!!";

    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "Before WriteResult!!!!";

    // write result
    WriteResult(imgPath, &outputs, output_path);


    return APP_ERR_OK;
}


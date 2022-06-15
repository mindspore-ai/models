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

#include "DGCN.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"


APP_ERROR DgcnBase::Init(const InitParam &initParam) {
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

APP_ERROR DgcnBase::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR DgcnBase::ReadTensorFromFile(const std::string &file, float *data, uint32_t size) {
    if (data == NULL) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    // open file
    std::ifstream in(file, std::ios::in | std::ios::binary);
    // check file validity
    if (in.fail()) {
        LogError << "Failed to open label file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    in.read(reinterpret_cast<char*>(data), sizeof(float) * size);
    return APP_ERR_OK;
}

APP_ERROR DgcnBase::ReadInputTensor(const std::string &fileName, uint32_t index,
                                        std::vector<MxBase::TensorBase> *inputs, uint32_t size,
                                        MxBase::TensorDataType type, const std::string &dataname) {
    float* data = new float[size];
    APP_ERROR ret = ReadTensorFromFile(fileName, data, size);
    std::cout << "size is" << size << std::endl;
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }
    const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;
    LogInfo << "dataSize: " << dataSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }
    uint32_t length4 = 1;
    if (dataname == "cora") {
              length4 = 2708;
    }
    if (dataname == "pubmed") {
              length4 = 19717;
    }
    if (dataname == "citeseer") {
              length4 = 3312;
    }
    std::vector<uint32_t> shape = {length4, size/length4};
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, type));
    std::cout << "shape con" << shape[0] << shape[1] << std::endl;
    return APP_ERR_OK;
}

APP_ERROR DgcnBase::Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs) {
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

APP_ERROR DgcnBase::PostProcess(std::vector<MxBase::TensorBase> *outputs, std::vector<float> *result) {
    MxBase::TensorBase &tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // check tensor is available
    auto outputShape = tensor.GetShape();
    uint32_t length = outputShape[0];
    uint32_t classNum = outputShape[1];
    LogInfo << "output shape is: " << length << " "<< classNum << " " << std::endl;
    void* data = tensor.GetBuffer();
    LogInfo << "output shape is: " << data;
    for (uint32_t i = 0; i < length; i++) {
        for (uint32_t j = 0; j < classNum; j++) {
            float value = *(reinterpret_cast<float*>(data) + i * classNum + j);
            result->push_back(value);
       }
    }
    return APP_ERR_OK;
}

APP_ERROR DgcnBase::getAucResult(std::vector<float> *result, uint32_t nice, const std::string &dataname) {
    if (dataname == "pubmed") {
    std::ofstream fp_r("pred3.txt", std::ofstream::app);
    int k = 0;
    for (uint32_t i = 0; i < 59151; i++) {
            fp_r << result->at(i) << " ";
            k++;
            if (k != 0&&k%3 == 0) {
               fp_r << std::endl;
        }
    }
    fp_r << std::endl;
    fp_r.close();
    }
    if (dataname == "cora") {
    std::ofstream fp_r("pred1.txt", std::ofstream::app);
    int k = 0;
    for (uint32_t i = 0; i < 18956; i++) {
            fp_r << result->at(i) << " ";
            k++;
            if (k != 0&&k%7 == 0) {
               fp_r << std::endl;
            }
    }
    fp_r << std::endl;
    fp_r.close();
    }
    if (dataname == "citeseer") {
    std::ofstream fp_r("pred2.txt", std::ofstream::app);
    int k = 0;
    for (uint32_t i = 0; i < 19962; i++) {
            fp_r << result->at(i) << " ";
            k++;
            if (k != 0&&k%6 == 0) {
               fp_r << std::endl;
            }
    }
    fp_r << std::endl;
    fp_r.close();
    }
    return APP_ERR_OK;
}


APP_ERROR DgcnBase::Process(const std::string &inferPath, const std::string &fileName, const std::string &dataname) {
    std::vector<MxBase::TensorBase> inputs = {};
    uint32_t lengtht1 = 0;
    uint32_t lengtht2 = 0;
    uint32_t lengtht3 = 0;
    if (dataname == "cora") {
     lengtht1 = 7333264;
     lengtht2 = 7333264;
     lengtht3 = 3880564;
    }
    if (dataname == "citeseer") {
     lengtht1 = 10969344;
     lengtht2 = 10969344;
     lengtht3 = 12264336;
    }
    if (dataname == "pubmed") {
     lengtht1 = 388760089;
     lengtht2 = 388760089;
     lengtht3 = 9858500;
    }
    std::string filename1 = "diffusions.bin";
    std::string spareFile = inferPath + "/00_data/" + filename1;
    LogInfo << "read file name: " << spareFile;
    APP_ERROR ret = ReadInputTensor(spareFile, 0, &inputs, lengtht1, MxBase::TENSOR_DTYPE_FLOAT32, dataname);
    if (ret != APP_ERR_OK) {
        LogError << "Read input spare failed, ret=" << ret << ".";
        return ret;
    }
    std::string filename2 = "ppmi.bin";
    std::string denseFile = inferPath + "/01_data/" + filename2;
    LogInfo << "read file name: " << denseFile;
    ret = ReadInputTensor(denseFile, 1, &inputs, lengtht2, MxBase::TENSOR_DTYPE_FLOAT32, dataname);
    if (ret != APP_ERR_OK) {
        LogError << "Read input dense file failed, ret=" << ret << ".";
        return ret;
    }
    std::string filename3 = "feature.bin";
    std::string labelFile = inferPath + "/02_data/" + filename3;
    LogInfo << "read file name: " << labelFile;
    ret = ReadInputTensor(labelFile, 2, &inputs, lengtht3, MxBase::TENSOR_DTYPE_FLOAT32, dataname);
    if (ret != APP_ERR_OK) {
        LogError << "Read label file failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> outputs = {};
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<float> result;
    ret = PostProcess(&outputs, &result);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    uint32_t nice = outputs.size();
    std::cout << "nice is" << nice << std::endl;

    ret = getAucResult(&result, nice, dataname);
    if (ret != APP_ERR_OK) {
        LogError << "CalcF1Score read label failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

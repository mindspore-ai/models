/*
* Copyright(C) 2022. Huawei Technologies Co.,Ltd
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
#include "c3dInfer.h"
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>
#include <string>
#include <map>
#include <fstream>
#include <memory>
#include <vector>
#include <typeinfo>
#include <cstdio>
#include <iostream>
#include <sstream>
#include "opencv2/opencv.hpp"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"


c3dInfer::c3dInfer(const uint32_t &deviceId, const std::string &modelPath) : deviceId_(deviceId) {
    LogInfo << "c3dInfer Construct!!!";
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
    LogInfo << "c3dInfer Construct End!!!";
}

c3dInfer::~c3dInfer() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
}

// model infer
APP_ERROR c3dInfer::Inference(const std::vector<MxBase::TensorBase> &inputs,
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

    MxBase::TensorBase tens = inputs.at(0);
    auto inputShape = tens.GetShape();
    LogInfo << "input shape is: " << inputShape[0] << " " << inputShape[1]
        << " "<< inputShape[2] << " " << inputShape[3]<< " " << inputShape[4] << std::endl;
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

// read csv files
float *csvRead(std::string filename) {
    float *result = new float[112*112*3];
    std::ifstream infile(filename, std::ios_base::in);
    std::string line;
    int sizex = 112;
    int ind = 0;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string str;
        for (int i = 0; i < sizex; i++) {
            std::getline(ss, str, ',');
            result[ind] = stod(str);
            ind = ind+1;
        }
    }
    return result;
}

APP_ERROR c3dInfer::Process(const std::string &imgPath) {
    LogInfo << "READ File!!!!";
    uint32_t MAX_LENGTH = 1*3*16*112*112;

    // read file into inputs
    float *data = new float[MAX_LENGTH];
    float *imdata;
    std::ifstream infile("./data/csvfilename.csv");
    std::string line;
    int per_in = 0;
    int input_num = 0;
    int acc_num = 0;
    clock_t start, end;
    start = clock();
    while (std::getline(infile, line)) {
        std::istringstream sin(line);
        std::vector<std::string> fields;
        std::string field;
        while (getline(sin, field, ',')) {
            fields.push_back(field);
        }
        int Class_n = stoi(fields[0]);
        std::string project_name = fields[1];
        project_name = imgPath + project_name;
        if (per_in < 15) {
            imdata = csvRead(project_name);
            for (int j = 0; j < 112*112; j++) {
                data[j+112*112*per_in] = imdata[j];
                data[j+112*112*per_in+112*112*16] = imdata[j+112*112];
                data[j+112*112*per_in+112*112*16*2] = imdata[j+112*112*2];
            }
            per_in = per_in + 1;
            continue;
        }
        imdata = csvRead(project_name);
        for (int k = 0; k < 112*112; k++) {
            data[k+112*112*15] = imdata[k];
            data[k+112*112*15+112*112*16] = imdata[k+112*112];
            data[k+112*112*15+112*112*16*2] = imdata[k+112*112*2];
        }
        per_in = 0;

        std::vector<MxBase::TensorBase> inputs = {};
        const uint32_t dataSize = MAX_LENGTH*4;
        LogError << "DataSize: " << dataSize << ".";
        MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
        MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize,
            MxBase::MemoryData::MEMORY_HOST_MALLOC);
        APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Memory malloc and copy failed.";
            return ret;
        }
        std::vector<uint32_t> shape = {1, 3, 16, 112, 112};
        inputs.push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32));

        std::vector<MxBase::TensorBase> outputs = {};
        LogInfo << "Before Inference!!!!!";

        ret = Inference(inputs, &outputs);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }

        MxBase::TensorBase &tensor = outputs.at(0);
        ret = tensor.ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Tensor deploy to host failed.";
            return ret;
        }

        // check tensor is available
        auto outputShape = tensor.GetShape();
        LogInfo << "output shape is: " << outputShape[0] << " " << outputShape[1] << std::endl;
        void* outdata = tensor.GetBuffer();
        float* outval = reinterpret_cast<float*>(outdata);
        float maxv = 0;
        int C = 0;
        for (int i = 0; i < 101; i++) {
            if (outval[i] > maxv) {
                maxv = outval[i];
                C = i;
            }
        }
        if (C == Class_n) {
            acc_num = acc_num + 1;
        }
        input_num = input_num + 1;
        LogInfo << "Classes of predictions is: " << C << ". ";
        LogInfo << "Correct label is: " << Class_n << ". ";
        LogInfo << "Accuracy is :  " << float(acc_num)/float(input_num) << std::endl;

        // write result
        std::ofstream outFile;
        outFile.open("./result.csv", std::ios_base::app);
        outFile << Class_n << ',' << C << ',' << float(acc_num)/float(input_num) << std::endl;
        outFile.close();
    }
    end = clock();
    LogInfo << "Number of predictions is :  " << input_num << std::endl;
    LogInfo << "Number of correct predictions is :  " << acc_num << std::endl;
    LogInfo << "Inference time = " << double(end-start)/CLOCKS_PER_SEC << "s" << std::endl;
    return APP_ERR_OK;
}

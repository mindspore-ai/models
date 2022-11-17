/*
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
 * ============================================================================
 */

#include "Mcnn.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include <opencv2/opencv.hpp>


using MxBase::TensorDesc;
using MxBase::TensorBase;
using MxBase::MemoryData;
using MxBase::MemoryHelper;
using MxBase::TENSOR_DTYPE_FLOAT32;
using MxBase::DynamicInfo;
using MxBase::DynamicType;

void PrintTensorShape(const std::vector<MxBase::TensorDesc> &tensorDescVec, const std::string &tensorName) {
    LogInfo << "The shape of " << tensorName << " is as follows:";
    for (size_t i = 0; i < tensorDescVec.size(); ++i) {
        LogInfo << "  Tensor " << i << ":";
        for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
            LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
        }
    }
}

void PrintInputShape(const std::vector<MxBase::TensorBase> &input) {
    MxBase::TensorBase img = input[0];
    LogInfo << "  -------------------------input0 ";
    LogInfo << img.GetDataType();
    LogInfo << img.GetShape()[0] << ", " << img.GetShape()[1] \
    << ", "  << img.GetShape()[2] << ", " << img.GetShape()[3];
    LogInfo << img.GetSize();
}

APP_ERROR Mcnn::Init(const InitParam &initParam) {
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
    srPath_ = initParam.srPath;
    gtPath_ = initParam.gtPath;
    PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
    PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");


    return APP_ERR_OK;
}

APP_ERROR Mcnn::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();

    MxBase::DeviceManager::GetInstance()->DestroyDevices();

    return APP_ERR_OK;
}


APP_ERROR Mcnn::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    *imageMat = cv::imread(imgPath, 0);
    return APP_ERR_OK;
}


std::string Trim(const std::string &str) {
    std::string str_new = str;
    str_new.erase(0, str.find_first_not_of(" \t\r\n"));
    str_new.erase(str.find_last_not_of(" \t\r\n") + 1);
    return str_new;
}


float ReadCsv(std::string csvName) {
    std::ifstream fin(csvName);
    std::string line;
    float num = 0;
    while (getline(fin, line)) {
        std::istringstream sin(line);
        std::vector<std::string> fields;
        std::string field;
        int len = 0;
        while (getline(sin, field, ',')) {
            len++;
            fields.push_back(field);
        }
        for (int i = 0; i < len; i++) {
            std::string name = Trim(fields[i]);
            float num_float = std::stof(name);
            num = num + num_float;
        }
    }
    return num;
}


APP_ERROR Mcnn::PadImage(const cv::Mat &imageMat, cv::Mat *imgPad) {
    size_t W_o = imageMat.cols, H_o = imageMat.rows;
    size_t W_b = 512 - W_o / 2;
    size_t H_b = 512 - H_o / 2;
    size_t W_e = W_b + W_o;
    size_t H_e = H_b + H_o;
    for (size_t h = 0; h < 1024; h++) {
        for (size_t w = 0; w < 1024; w++) {
            if (H_b <= h && h < H_e && W_b <= w && w < W_e) {
                imgPad->at<uchar>(h, w) = imageMat.at<uchar>(h - H_b, w - W_b);
            } else {
                imgPad->at<uchar>(h, w) = 0;
            }
        }
    }
    return APP_ERR_OK;
}


APP_ERROR Mcnn::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector <uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t) modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }
    APP_ERROR ret = PadImage(imageMat, &imgPad_);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Img pad error";
        return ret;
    }
    // mat NHWC to NCHW
    size_t H = 1024, W = 1024, C = 1;
    LogInfo << "dataSize:" << dataSize;
    dataSize = dataSize * 4;
    int id;

    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                id = (C - c - 1) * (H * W) + h * W + w;
                mat_data_[id] = static_cast<float>(imgPad_.at<uchar>(h, w));
            }
        }
    }

    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(reinterpret_cast<void *>(&mat_data_[0]), dataSize, MemoryData::MEMORY_HOST_MALLOC);
    ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector <uint32_t> shape = {1, 1, 1024, 1024};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}


APP_ERROR Mcnn::Inference(const std::vector<MxBase::TensorBase> &inputs,
                           std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    dynamicInfo.batchSize = 1;

    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR Mcnn::PostProcess(std::vector<MxBase::TensorBase> outputs, const std::string &imgName) {
    LogInfo << "output_size:" << outputs.size();
    LogInfo <<  "output0_datatype:" << outputs[0].GetDataType();
    LogInfo << "output0_shape:" << outputs[0].GetShape()[0] << ", " \
    << outputs[0].GetShape()[1] << ", "  << outputs[0].GetShape()[2] << ", " << outputs[0].GetShape()[3];
    LogInfo << "output0_bytesize:"  << outputs[0].GetByteSize();

    APP_ERROR ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
    }
    float *outputPtr = reinterpret_cast<float *>(outputs[0].GetBuffer());

    size_t  H = 1024/4 , W = 1024/4 , C = 1;

    float tmpNum;
    float pre = 0;
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                tmpNum = static_cast<float>(*(outputPtr + (C - c - 1) * (H * W) + h * W + w));
                pre = pre + tmpNum;
            }
        }
    }
    std::string imgName2 = imgName;
    int len = imgName.length();
    imgName2[len-3] = 'c';
    imgName2[len-2] = 's';
    imgName2[len-1] = 'v';
    std::string gtName = gtPath_;
    gtName.append(imgName2);
    LogInfo << gtName;
    LogInfo << "pre:" << pre;
    float gt_count = ReadCsv(gtName);
    LogInfo << "gt:" << gt_count;
    mae += fabs(gt_count - pre);
    mse = mse+ (gt_count - pre)*(gt_count - pre);
    LogInfo << "mae:" << fabs(gt_count - pre);
    return APP_ERR_OK;
}


APP_ERROR Mcnn::Process(const std::string &imgPath, const std::string &imgName) {
    LogInfo << imgName;
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, &imageMat);

    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    size_t o_img_W = imageMat.cols, o_img_H = imageMat.rows, o_img_C = imageMat.channels();
    LogInfo << o_img_C << "," << o_img_H << "," << o_img_W;


    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, &tensorBase);
    cv::imwrite(srPath_ + "/" + imgName, imgPad_);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);

    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    ret = PostProcess(outputs, imgName);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

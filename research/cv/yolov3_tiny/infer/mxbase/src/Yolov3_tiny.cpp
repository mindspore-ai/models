/*
 * Copyright (c) 2022 Huawei Technologies Co., Ltd.
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

#include <dirent.h>
#include <unistd.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "MxBase/Log/Log.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "Yolov3_tiny.h"

namespace {
    const uint32_t MODEL_HEIGHT = 640;
    const uint32_t MODEL_WIDTH = 640;
    const uint32_t MODEL_CHANNEL = 12;
    const uint32_t BATCH_NUM = 1;
    const std::vector<float> MEAN = {0.485, 0.456, 0.406};
    const std::vector<float> STD = {0.229, 0.224, 0.225};
    const float MODEL_MAX = 255.0;
    const int DATA_SIZE = 1228800;
    const int coco_class_nameid[80] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                       22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                       46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                                       67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
}  // namespace

void getfilename(std::string *filename, std::string *filedir, const std::string &imgpath);

APP_ERROR Yolov3_tiny::Init(const InitParam &initParam) {
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

APP_ERROR Yolov3_tiny::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Yolov3_tiny::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR Yolov3_tiny::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    cv::resize(srcImageMat, dstImageMat, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    return APP_ERR_OK;
}

APP_ERROR Yolov3_tiny::WhcToChw(const cv::Mat &srcImageMat, std::vector<float> *imgData) {
    int channel = srcImageMat.channels();
    std::vector<cv::Mat> bgrChannels(channel);
    cv::split(srcImageMat, bgrChannels);
    for (int i = channel - 1; i >= 0; i--) {
        std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
        std::transform(data.begin(), data.end(), data.begin(),
                       [&](float item) {return ((item / MODEL_MAX - MEAN[channel - i - 1]) / STD[channel - i - 1]); });
        imgData->insert(imgData->end(), data.begin(), data.end());
    }
    return APP_ERR_OK;
}

APP_ERROR Yolov3_tiny::ArrayToTensorBase(const std::vector<float> &imgData, MxBase::TensorBase &tensorBase) {
    float data[IMG_C * IMG_H * IMG_W];
    for (int i = 0; i < IMG_C * IMG_H * IMG_W; i++) {
        data[i] = imgData[i];
    }
    const uint32_t dataSize = IMG_C * IMG_H * IMG_W * sizeof(float);
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, 3, 640, 640};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR Yolov3_tiny::Inference(const std::vector<MxBase::TensorBase> &inputs, \
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
    dynamicInfo.batchSize = 1;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR Yolov3_tiny::SaveResult(MxBase::TensorBase *tensor_0, MxBase::TensorBase *tensor_1,
                                  const std::string &resultpath_0, const std::string &resultpath_1) {
    std::ofstream outfile(resultpath_0, std::ios::binary);
    APP_ERROR ret = (*tensor_0).ToHost();
    if (ret != APP_ERR_OK) {
        LogError << "ToHost failed";
        return ret;
    }
    if (outfile.fail()) {
        LogError << "Failed to open result file0: ";
        return APP_ERR_COMM_FAILURE;
    }
    outfile.write(reinterpret_cast<char *>((*tensor_0).GetBuffer()), sizeof(float) * Output_small);
    outfile.close();

    std::ofstream outfile_1(resultpath_1, std::ios::binary);
    APP_ERROR ret_1 = (*tensor_1).ToHost();
    if (ret_1 != APP_ERR_OK) {
        LogError << "ToHost failed";
        return ret_1;
    }
    if (outfile_1.fail()) {
        LogError << "Failed to open result file0: ";
        return APP_ERR_COMM_FAILURE;
    }
    outfile_1.write(reinterpret_cast<char *>((*tensor_1).GetBuffer()), sizeof(float) * Output_big);
    outfile_1.close();

    return APP_ERR_OK;
}


APP_ERROR Yolov3_tiny::Process(const std::string &imgPath, const std::string &resultPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    ret = ResizeImage(imageMat, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ResizeImage failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<float> imgData;
    ret = WhcToChw(imageMat, &imgData);
    if (ret != APP_ERR_OK) {
        LogError << "WhcToChw failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    MxBase::TensorBase tensorBase;
    ret = ArrayToTensorBase(imgData, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "ArrayToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);

    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::string filename = "";
    std::string filedir = "";
    getfilename(&filename, &filedir, imgPath);
    std::string resultpath_0 = resultPath + "/" + filename + "_0.bin";
    std::string resultpath_1 = resultPath + "/" + filename + "_1.bin";
    std::string resultdir = resultPath + "/" + filedir;

    DIR *dirPtr = opendir(resultdir.c_str());
    if (dirPtr == nullptr) {
        std::string sys = "mkdir -p "+ resultdir;
        system(sys.c_str());
    }

    ret = SaveResult(&outputs[0], &outputs[1], resultpath_0, resultpath_1);
    if (ret != APP_ERR_OK) {
        LogError << "SaveResult failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}


void getfilename(std::string *filename, std::string *filedir, const std::string &imgpath) {
    int i, j = 0, count = 0;
    for (i = imgpath.length() - 1; i >= 0; i--) {
        // '/' is the delimiter between the file name and the parent directory in imgpath
        if (imgpath[i] == '/') {
            count++;
            if (count == 2) {
                j = i;
                break;
            }
        }
    }
    // '.' is the delimiter between the file name and the file suffix
    while (imgpath[++j] != '.') {
        *filename += imgpath[j];
    }

    //'/' is the delimiter between the file name and the file directory
    j = i;
    while (imgpath[++j] != '/') {
        *filedir += imgpath[j];
    }
}

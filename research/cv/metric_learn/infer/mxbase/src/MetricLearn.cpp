/*
 * Copyright (c) 2022 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "MetricLearn.h"

void getfilename(std::string *filename, std::string *filedir, const std::string &imgpath);

APP_ERROR MetricLearn::Init(const InitParam &initParam) {
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

APP_ERROR MetricLearn::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR MetricLearn::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR MetricLearn::ResizeShortImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    int height = srcImageMat.rows;
    int width = srcImageMat.cols;
    float percent = static_cast<float>(224.0) / std::min(height, width);
    int INTER_LANCZOS4 = 4;
    cv::resize(srcImageMat, dstImageMat, cv::Size(round(width * percent),
    round(height * percent)), 0, 0, INTER_LANCZOS4);
    return APP_ERR_OK;
}

APP_ERROR MetricLearn::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    static constexpr uint32_t resizeHeight = 224;
    static constexpr uint32_t resizeWidth = 224;
    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR MetricLearn::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize =  imageMat.cols *  imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR MetricLearn::Inference(const std::vector<MxBase::TensorBase> &inputs, \
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

APP_ERROR MetricLearn::SaveResult(MxBase::TensorBase *tensor, const std::string &resultpath) {
    std::ofstream outfile(resultpath, std::ios::binary);
    APP_ERROR ret = (*tensor).ToHost();
    if (ret != APP_ERR_OK) {
        LogError << "ToHost failed";
        return ret;
    }
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }
    outfile.write(reinterpret_cast<char *>((*tensor).GetBuffer()), sizeof(float) * FEATURE_NUM);
    outfile.close();

    return APP_ERR_OK;
}


APP_ERROR MetricLearn::Process(const std::string &imgPath, const std::string &resultPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    ret = ResizeShortImage(imageMat, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ResizeShortImage failed, ret=" << ret << ".";
        return ret;
    }

    ret = ResizeImage(imageMat, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ResizeImage failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
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
    std::string resultpath = resultPath + "/" + filename + ".bin";
    std::string resultdir = resultPath + "/" + filedir;

    DIR *dirPtr = opendir(resultdir.c_str());
    if (dirPtr == nullptr) {
        std::string sys = "mkdir -p "+ resultdir;
        system(sys.c_str());
    }

    ret = SaveResult(&outputs[0], resultpath);
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

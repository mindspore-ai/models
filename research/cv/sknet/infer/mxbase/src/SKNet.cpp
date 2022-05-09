/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "SKNet.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include <opencv2/dnn.hpp>
#include "MxBase/Log/Log.h"

using  namespace MxBase;
using  namespace cv::dnn;
namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
    const uint32_t BATCH_SIZE = 32;
}

APP_ERROR SKNet::Init(const InitParam &initParam) {
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
    MxBase::ConfigData configData;
    const std::string softmax = initParam.softmax ? "true" : "false";

    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("TOP_K", std::to_string(initParam.topk));
    configData.SetJsonValue("SOFTMAX", softmax);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::Resnet50PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR SKNet::DeInit() {
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR SKNet::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR SKNet::ProcessImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    static constexpr uint32_t resizeHeight = 224;
    static constexpr uint32_t resizeWidth = 224;
    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    cv::Scalar mean = cv::Scalar(0.4914, 0.4822, 0.4465);
    cv::Scalar std = cv::Scalar(0.2023, 0.1994, 0.2010);
    dstImageMat = dstImageMat / 255;
    dstImageMat = (dstImageMat - mean) / std;
    return APP_ERR_OK;
}

APP_ERROR SKNet::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    std::vector<float> dst_data;
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(imageMat, bgrChannels);
    for (unsigned int i = 0; i < bgrChannels.size(); i++) {
        std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
        dst_data.insert(dst_data.end(), data.begin(), data.end());
    }
    float *buffer = new float[dst_data.size()];
    if (!dst_data.empty()) {
        memcpy(buffer, &dst_data[0], dst_data.size()*sizeof(float));
    }
    const uint32_t dataSize =  3 * 224 * 224 * 4;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(buffer, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {3, 224, 224};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR SKNet::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR SKNet::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                        std::vector<std::vector<MxBase::ClassInfo>> *clsInfos) {
    APP_ERROR ret = post_->Process(inputs, *clsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR SKNet::SaveResult(const std::vector<std::string> &batchImgPaths,
                            const std::vector<std::vector<MxBase::ClassInfo>> &batchClsInfos) {
    uint32_t batchIndex = 0;
    for (auto &imgPath : batchImgPaths) {
        std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
        size_t dot = fileName.find_last_of(".");
        std::string resFileName = "./results/" + fileName.substr(0, dot) + ".txt";
        std::ofstream outfile(resFileName);
        if (outfile.fail()) {
            LogError << "Failed to open result file: ";
            return APP_ERR_COMM_FAILURE;
        }
        auto clsInfos = batchClsInfos[batchIndex];
            std::string resultStr;
            for (auto clsInfo : clsInfos) {
                LogDebug << " className:" << clsInfo.className << " confidence:" << clsInfo.confidence <<
                " classIndex:" <<  clsInfo.classId;
                resultStr += std::to_string(clsInfo.classId) + " ";
            }
            outfile << resultStr << std::endl;
            batchIndex++;
        outfile.close();
    }
    return APP_ERR_OK;
}

APP_ERROR SKNet::Process(const std::vector<std::string> &batchImgPaths) {
    APP_ERROR ret = APP_ERR_OK;
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    std::vector<MxBase::TensorBase> batchImgMat = {};
    for (auto &imgPath : batchImgPaths) {
        cv::Mat imageMat;
        ret = ReadImage(imgPath, &imageMat);
        if (ret != APP_ERR_OK) {
            LogError << "ReadImage failed, ret=" << ret << ".";
            return ret;
        }
        imageMat.convertTo(imageMat, CV_32F);
        ret = ProcessImage(imageMat, imageMat);
        if (ret != APP_ERR_OK) {
            LogError << "ProcessImage failed, ret=" << ret << ".";
            return ret;
        }
        MxBase::TensorBase tensorBase;
        ret = CVMatToTensorBase(imageMat, tensorBase);
        if (ret != APP_ERR_OK) {
            LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
            return ret;
        }

        batchImgMat.push_back(tensorBase);
    }
    MxBase::TensorBase tensorBase;
    tensorBase.BatchStack(batchImgMat, tensorBase);
    inputs.push_back(tensorBase);
    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<std::vector<MxBase::ClassInfo>> BatchClsInfos = {};
    ret = PostProcess(outputs, &BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    ret = SaveResult(batchImgPaths, BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

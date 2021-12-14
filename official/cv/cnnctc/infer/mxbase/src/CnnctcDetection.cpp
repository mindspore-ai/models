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
 */

#include "CnnctcDetection.h"
#include <unistd.h>
#include <sys/stat.h>
#include <utility>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR CnnctcDetection::Init(const InitParam &initParam) {
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
    const std::string argmax = initParam.argmax ? "true" : "false";
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("OBJECT_NUM", std::to_string(initParam.objectNum));
    configData.SetJsonValue("BLANK_INDEX", std::to_string(initParam.blankIndex));
    configData.SetJsonValue("SOFTMAX", softmax);
    configData.SetJsonValue("WITH_ARGMAX", argmax);
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);

    post_ = std::make_shared<MxBase::CrnnPostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "CrnnPostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR CnnctcDetection::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR CnnctcDetection::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR CnnctcDetection::ResizeAndPad(const cv::Mat &srcImageMat, MxBase::TensorBase *tensorBase)  {
    uint32_t imgH = 32;
    uint32_t imgW = 100;
    uint32_t resizeWidth;
    float ratio = srcImageMat.cols / static_cast<float>(srcImageMat.rows);
    if (std::ceil(imgH * ratio) > imgW) {
        resizeWidth = imgW;
    } else {
        resizeWidth = std::ceil(imgH * ratio);
    }
    cv::Mat dstImageMat;
    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, imgH), cv::INTER_CUBIC);
    if (resizeWidth != imgW) {
        cv::copyMakeBorder(dstImageMat, dstImageMat, 0, 0, 0, imgW - resizeWidth,
            cv::BorderTypes::BORDER_REPLICATE);
    }
    cv::imwrite("tmp.jpg", dstImageMat);
    CVMatToTensorBase(dstImageMat, tensorBase);
    return APP_ERR_OK;
}

APP_ERROR CnnctcDetection::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
    const uint32_t YUV444_RGB_WIDTH_NU = 3;
    const uint32_t dataSize = imageMat.cols * imageMat.rows * YUV444_RGB_WIDTH_NU;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {imageMat.rows * YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR CnnctcDetection::Inference(const std::vector<MxBase::TensorBase> &inputs,
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

APP_ERROR CnnctcDetection::PostProcess(std::vector<MxBase::TensorBase> *outputs,
            std::vector<MxBase::TextsInfo> *textInfos) {
    APP_ERROR ret = post_->Process(*outputs, *textInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR CnnctcDetection::Process(const std::string &imgPath, std::string *result) {
    // process image
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, &imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    ret = ResizeAndPad(imageMat, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "ResizeAndPad failed, ret=" << ret << ".";
        return ret;
    }

    ret = ReadImage("tmp.jpg", &imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    ret = CVMatToTensorBase(imageMat, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TextsInfo> TextInfos = {};
    ret = PostProcess(&outputs, &TextInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    imageMat.release();
    uint32_t topkIndex = 1;
    for (auto textInfos : TextInfos) {
        if (topkIndex > 1) {
            break;
        }
        for (size_t i = 0; i < textInfos.text.size(); ++i) {
            LogDebug << " top" << topkIndex << " text: " << textInfos.text[i];
            *result = textInfos.text[i];
        }
        topkIndex++;
    }
    return APP_ERR_OK;
}

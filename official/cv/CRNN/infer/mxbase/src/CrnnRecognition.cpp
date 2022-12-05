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

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "CrnnRecognition.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"


APP_ERROR CrnnRecognition::Init(const InitParam &initParam) {
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

APP_ERROR CrnnRecognition::DeInit() {
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR CrnnRecognition::ReadAndResize(const std::string &imgPath, MxBase::TensorBase &outputTensor) {
    cv::Mat srcImageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    LogInfo << "imgPath" << imgPath;
    cv::Mat dstImageMat;
    uint32_t resizeWidth = 100;
    uint32_t resizeHeight = 32;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));

    uint32_t dataSize = dstImageMat.cols * dstImageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;

    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(dstImageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {dstImageMat.rows * MxBase::YUV444_RGB_WIDTH_NU,
                static_cast<uint32_t>(dstImageMat.cols)};
    outputTensor = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR CrnnRecognition::Inference(const std::vector<MxBase::TensorBase> &inputs,
            std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }

        int tmp_shape = 0;
        tmp_shape = shape[1];
        shape[1] = shape[0];
        shape[0] = tmp_shape;

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

    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;

    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR CrnnRecognition::PostProcess(const std::vector<MxBase::TensorBase>& tensors,
            std::vector<MxBase::TextsInfo>& textInfos) {
    APP_ERROR ret = post_->Process(tensors, textInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR CrnnRecognition::Process(const std::string &imgPath, std::string &result) {
    MxBase::TensorBase resizeImage;
    APP_ERROR ret = ReadAndResize(imgPath, resizeImage);
    if (ret != APP_ERR_OK) {
        LogError << "Read and resize image failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(resizeImage);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TextsInfo> TextInfos = {};
    ret = PostProcess(outputs, TextInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    uint32_t topkIndex = 1;
    for (auto textInfos : TextInfos) {
        if (topkIndex > 1) {
            break;
        }
        for (size_t i = 0; i < textInfos.text.size(); ++i) {
            LogDebug << " top" << topkIndex << " text: " << textInfos.text[i];
            result = textInfos.text[i];
        }
        topkIndex++;
    }
    return APP_ERR_OK;
}

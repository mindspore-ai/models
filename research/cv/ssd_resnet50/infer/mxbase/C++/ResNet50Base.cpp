/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

// 主代码逻辑
#include "SSDResNet50.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

using MxBase::TensorBase;
using MxBase::ObjectInfo;
using MxBase::ResizedImageInfo;
using MxBase::DeviceManager;
using MxBase::TensorContext;
using MxBase::DvppWrapper;
using MxBase::ModelInferenceProcessor;
using MxBase::ConfigData;
using MxBase::SsdMobilenetFpnMindsporePost;
using MxBase::YUV444_RGB_WIDTH_NU;
using MxBase::MemoryData;
using MxBase::MemoryHelper;
using MxBase::TENSOR_DTYPE_UINT8;
using MxBase::DynamicInfo;
using MxBase::DynamicType;
using MxBase::RESIZER_STRETCHING;

APP_ERROR SSDResNet50::Init(const InitParam &initParam) {
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
    MxBase::ConfigData configData;
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("SCORE_THRESH", std::to_string(initParam.score_thresh));
    configData.SetJsonValue("IOU_THRESH", std::to_string(initParam.iou_thresh));
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::SsdMobilenetFpnMindsporePost>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "SSDResNet50 init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR SSDResNet50::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR SSDResNet50::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR SSDResNet50::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    static constexpr uint32_t resizeHeight = 640;
    static constexpr uint32_t resizeWidth = 640;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR SSDResNet50::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize =  imageMat.cols *  imageMat.rows * YUV444_RGB_WIDTH_NU;
    LogInfo << "image size after crop" << imageMat.cols << " " << imageMat.rows;
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(imageMat.data, dataSize, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {imageMat.rows * YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR SSDResNet50::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                       std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t) modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    dynamicInfo.batchSize = 1;

    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR SSDResNet50::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                         std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
                                         const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                                         const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
    APP_ERROR ret = post_->Process(inputs, objectInfos, resizedImageInfos, configParamMap);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR SSDResNet50::SaveResult(const std::string &imgPath,
                            std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos) {
    std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = fileName.find_last_of(".");
    std::string resFileName = "./results_" + fileName.substr(0, dot) + ".txt";
    std::ofstream outfile(resFileName);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }

    std::vector<ObjectInfo> objects = objectInfos.at(0);
    std::string resultStr;

    for (size_t i = 0; i < objects.size(); i++) {
        ObjectInfo obj = objects.at(i);
        std::string info = "BBox[" + std::to_string(i) + "]:[x0=" + std::to_string(obj.x0)
                + ", y0=" + std::to_string(obj.y0) + ", w=" + std::to_string(obj.x1 - obj.x0) + ", h="
                + std::to_string(obj.y1 - obj.y0) + "], confidence=" + std::to_string(obj.confidence)
                + ", classId=" + std::to_string(obj.classId) + ", className=" + obj.className;
        LogInfo << info;
        resultStr += info + "\n";
    }
    outfile << resultStr << std::endl;
    outfile.close();

    return APP_ERR_OK;
}

APP_ERROR SSDResNet50::Process(const std::string &imgPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);

    const uint32_t originHeight = imageMat.rows;
    const uint32_t originWidth = imageMat.cols;

    LogInfo << "image shape, size=" << originWidth << "," << originHeight << ".";
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    ResizeImage(imageMat, imageMat);

    TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, tensorBase);

    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(tensorBase);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "Inference success, ret=" << ret << ".";
    std::vector<MxBase::ResizedImageInfo> resizedImageInfos = {};

    ResizedImageInfo imgInfo;

    imgInfo.widthOriginal = originWidth;
    imgInfo.heightOriginal = originHeight;
    imgInfo.widthResize = 640;
    imgInfo.heightResize = 640;
    imgInfo.resizeType = MxBase::RESIZER_STRETCHING;

    resizedImageInfos.push_back(imgInfo);
    std::vector<std::vector<MxBase::ObjectInfo>> objectInfos = {};
    std::map<std::string, std::shared_ptr<void>> configParamMap = {};

    ret = PostProcess(outputs, objectInfos, resizedImageInfos, configParamMap);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    if (objectInfos.empty()) {
        LogInfo << "No object detected." << std::endl;
        return APP_ERR_OK;
    }

    ret = SaveResult(imgPath, objectInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

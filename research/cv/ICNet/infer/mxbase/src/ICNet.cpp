/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "ICNet.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

using std::max;
using std::vector;


APP_ERROR ICNet::Init(const InitParam &initParam) {
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
    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("MODEL_TYPE", std::to_string(initParam.modelType));
    configData.SetJsonValue("CHECK_MODEL", std::to_string(initParam.checkModel));
    configData.SetJsonValue("FRAMEWORK_TYPE", std::to_string(initParam.frameworkType));

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);
    post_ = std::make_shared<MxBase::Deeplabv3Post>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "Deeplabv3PostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR ICNet::DeInit() {
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR ICNet::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);
    return APP_ERR_OK;
}

APP_ERROR ICNet::Normalize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    constexpr size_t ALPHA_AND_BETA_SIZE = 3;
    cv::Mat float32Mat;
    srcImageMat.convertTo(float32Mat, CV_32FC3);

    std::vector<cv::Mat> tmp;
    cv::split(float32Mat, tmp);

    const std::vector<double> mean = {123.675, 116.28, 103.53};
    const std::vector<double> std = {58.395, 57.12, 57.375};
    for (size_t i = 0; i < ALPHA_AND_BETA_SIZE; ++i) {
        tmp[i].convertTo(tmp[i], CV_32FC3, 1 / std[i], - mean[i] / std[i]);
    }
    cv::merge(tmp, dstImageMat);
    return APP_ERR_OK;
}

void ICNet::GetResizeInfo(const cv::Mat &srcImageMat, MxBase::ResizedImageInfo &resizedImageInfo) {
    resizedImageInfo.heightOriginal = srcImageMat.rows;
    resizedImageInfo.heightResize = srcImageMat.rows;
    resizedImageInfo.widthOriginal = srcImageMat.cols;
    resizedImageInfo.widthResize = srcImageMat.cols;
    resizedImageInfo.resizeType = MxBase::RESIZER_MS_KEEP_ASPECT_RATIO;
}

APP_ERROR ICNet::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * imageMat.elemSize();
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {
        static_cast<uint32_t>(imageMat.rows),
        static_cast<uint32_t>(imageMat.cols),
        static_cast<uint32_t>(imageMat.channels())};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR ICNet::Inference(const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR ICNet::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::SemanticSegInfo> &segInfo, const std::vector<MxBase::ResizedImageInfo> &resizedInfo) {
    APP_ERROR ret = post_->Process(inputs, segInfo, resizedInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR ICNet::Process(const std::string &imgPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::ResizedImageInfo resizedImageInfo;
    GetResizeInfo(imageMat, resizedImageInfo);
    Normalize(imageMat, imageMat);

    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
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

    std::vector<MxBase::SemanticSegInfo> semanticSegInfos = {};
    std::vector<MxBase::ResizedImageInfo> resizedImageInfos = {resizedImageInfo};
    ret = PostProcess(outputs, semanticSegInfos, resizedImageInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    std::string resultPath = imgPath;
    size_t pos = resultPath.find_last_of(".");
    resultPath.replace(resultPath.begin() + pos, resultPath.end(), "_MxBase_color_infer.png");
    SaveResultToImage(semanticSegInfos[0], resultPath);
    return APP_ERR_OK;
}

void ICNet::SaveResultToImage(const MxBase::SemanticSegInfo &segInfo, const std::string &filePath) {
    cv::Mat imageMat(segInfo.pixels.size(), segInfo.pixels[0].size(), CV_8UC1);
    for (size_t x = 0; x < segInfo.pixels.size(); ++x) {
        for (size_t y = 0; y < segInfo.pixels[x].size(); ++y) {
            uint8_t gray = segInfo.pixels[x][y];
            imageMat.at<uchar>(x, y) = gray;
        }
    }

    cv::Mat imageGrayC3 = cv::Mat::zeros(imageMat.rows, imageMat.cols, CV_8UC3);
    const int chn = 3
    std::vector<cv::Mat> planes;
    for (int i = 0; i < chn; i++) {
        planes.push_back(imageMat);
    }
    cv::merge(planes, imageGrayC3);
    uchar rgbColorMap[256*3] = {
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        102, 102, 156,
        190, 153, 153,
        153, 153, 153,
        250, 170, 30,
        220, 220, 0,
        107, 142, 35,
        152, 251, 152,
        0, 130, 180,
        220, 20, 60,
        255, 0, 0,
        0, 0, 142,
        0, 0, 70,
        0, 60, 100,
        0, 80, 100,
        0, 0, 230,
        119, 11, 32,
    };
    cv::Mat lut(1, 256, CV_8UC3, rgbColorMap);

    cv::Mat imageColor;
    cv::LUT(imageGrayC3, lut, imageColor);
    cv::cvtColor(imageColor, imageColor, cv::COLOR_RGB2BGR);
    cv::imwrite(filePath, imageColor);
}

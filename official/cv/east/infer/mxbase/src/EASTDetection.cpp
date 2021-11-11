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

#include "EASTDetection.h"
#include <unistd.h>
#include <sys/stat.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"

namespace {
    const uint32_t MODEL_HEIGHT = 704;
    const uint32_t MODEL_WIDTH = 1280;
}

APP_ERROR EASTDetection::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;

    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogInfo << "Init devices failed, ret=" << ret << ".";
        return ret;
    }

    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogInfo << "Set context failed, ret=" << ret << ".";
    }

    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogInfo << "DvppWrapper init failed, ret=" << ret << ".";
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

    configData.SetJsonValue("NMS_THRESH", initParam.nmsThresh);
    configData.SetJsonValue("SCORE_THRESH", initParam.scoreThresh);
    configData.SetJsonValue("OUT_SIZE", std::to_string(initParam.outSize));
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);

    post_ = std::make_shared<MxBase::EASTPostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "CrnnPostProcess init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR EASTDetection::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR EASTDetection::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    (*imageMat) = cv::imread(imgPath, cv::IMREAD_COLOR);
    imageWidth_ = (*imageMat).cols;
    imageHeight_ = (*imageMat).rows;

    return APP_ERR_OK;
}

APP_ERROR EASTDetection::Resize(cv::Mat *srcImageMat, cv::Mat *dstImageMat) {
    cv::resize((*srcImageMat), (*dstImageMat), cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    return APP_ERR_OK;
}

APP_ERROR EASTDetection::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    (*tensorBase) = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR EASTDetection::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
        (*outputs).push_back(tensor);
    }

    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, (*outputs), dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR EASTDetection::PostProcess(const std::vector<MxBase::TensorBase>& tensors,
                                     std::vector<std::vector<MxBase::TextObjectInfo>> *textInfos) {
    MxBase::ResizedImageInfo imgInfo;
    imgInfo.widthOriginal = imageWidth_;
    imgInfo.heightOriginal = imageHeight_;
    imgInfo.widthResize = MODEL_WIDTH;
    imgInfo.heightResize = MODEL_HEIGHT;
    imgInfo.resizeType = MxBase::RESIZER_STRETCHING;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);

    APP_ERROR ret = post_->Process(tensors, (*textInfos), imageInfoVec);
    if (ret != APP_ERR_OK) {
        LogInfo << "Process failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR EASTDetection::WriteResult(const std::vector<std::vector<MxBase::TextObjectInfo>> &textInfos,
                                     const std::string &imgPath) {
    std::string file_name = imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = file_name.find_last_of(".");
    std::string resFileName = "../results/res_" + file_name.substr(0, dot) + ".txt";
    std::ofstream outfile(resFileName);
    if (outfile.fail()) {
        LogError << "Failed to open result file";
        return APP_ERR_COMM_FAILURE;
    }

    for (auto textInfo : textInfos) {
        for (auto info : textInfo) {
            std::string resultStr = "";
            resultStr += std::to_string(static_cast<int>(info.x0)) + "," +
                    std::to_string(static_cast<int>(info.y0)) + "," +
                    std::to_string(static_cast<int>(info.x1)) + "," + std::to_string(static_cast<int>(info.y1)) + "," +
                    std::to_string(static_cast<int>(info.x2)) + "," + std::to_string(static_cast<int>(info.y2)) + "," +
                    std::to_string(static_cast<int>(info.x3)) + "," + std::to_string(static_cast<int>(info.y3));
            outfile << resultStr << std::endl;
        }
    }

    outfile.close();

    return APP_ERR_OK;
}

APP_ERROR EASTDetection::Process(const std::string &imgPath) {
    // process image
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, &imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    ret = Resize(&imageMat, &imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
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

    std::vector<std::vector<MxBase::TextObjectInfo>> textInfos;
    ret = PostProcess(outputs, &textInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    for (auto objInfos : textInfos) {
        uint32_t topkIndex = 1;
        for (auto objInfo : objInfos) {
            LogDebug << "topkIndex:" << topkIndex << ", x0:" << objInfo.x0 << ", y0:" << objInfo.y0 << ", x1:"
                     << objInfo.x1 << ", y1:" << objInfo.y1 << ", x2:" << objInfo.x2 << ", y2:" << objInfo.y2
                     << ", x3:" << objInfo.x3 << ", y3:" << objInfo.y3 << ", confidence:" << objInfo.confidence;
            topkIndex++;
        }
    }

    ret = WriteResult(textInfos, imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "WriteResult failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

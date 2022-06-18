/*
 * Copyright 2022. Huawei Technologies Co., Ltd.
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
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "ResNeSt50Classify.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

// first function
APP_ERROR ResNeSt50Classify::Init(const InitParam &initParam) {
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
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("TOP_K", std::to_string(initParam.topk));
    configData.SetJsonValue("SOFTMAX", softmax);
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::Resnet50PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "Resnet50PostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    tfile_.open("mx_pred_result.txt");
    if (!tfile_) {
        LogError << "Open result file failed.";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    return APP_ERR_OK;
}

APP_ERROR ResNeSt50Classify::DeInit() {
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    tfile_.close();
    return APP_ERR_OK;
}

APP_ERROR ResNeSt50Classify::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);
    return APP_ERR_OK;
}

APP_ERROR ResNeSt50Classify::Resize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    cv::resize(srcImageMat, dstImageMat, cv::Size(320, 320));
    return APP_ERR_OK;
}

APP_ERROR ResNeSt50Classify::Crop(const cv::Mat &srcMat, cv::Mat &dstMat) {
    int height = srcMat.cols;
    int width = srcMat.rows;
    int out_width = 256;
    int out_height = 256;
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;
    left = static_cast<int>((width - out_width) / 2);
    right = static_cast<int>((width + out_width) / 2);
    top = static_cast<int>((height - out_height) / 2);
    bottom = static_cast<int>((height + out_height) / 2);

    static cv::Rect rectOfImg(top, left, bottom, right);
    dstMat = srcMat(rectOfImg).clone();

    return APP_ERR_OK;
}


APP_ERROR ResNeSt50Classify::Transe(const cv::Mat &srcMat, float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH]) {
    for (int i = 0; i < NET_HEIGHT; i++) {
        for (int j = 0; j < NET_WIDTH; j++) {
            cv::Vec3b nums = srcMat.at<cv::Vec3b>(i, j);
            imageArray[0][i][j] = nums[0];
            imageArray[1][i][j] = nums[1];
            imageArray[2][i][j] = nums[2];
        }
    }

    return APP_ERR_OK;
}

APP_ERROR ResNeSt50Classify::NormalizeImage(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH],
                                            float (&imageArrayNormal)[CHANNEL][NET_HEIGHT][NET_WIDTH]) {
    const std::vector<double> mean = {0.485, 0.456, 0.406};
    const std::vector<double> std = {0.229, 0.224, 0.225};
    for (int i = 0; i < CHANNEL; i++) {
        for (int j = 0; j < NET_HEIGHT; j++) {
            for (int k = 0; k < NET_WIDTH; k++) {
                imageArrayNormal[i][j][k] = imageArray[i][j][k] / 255;
            }
        }
    }
    for (int i = 0; i < NET_HEIGHT; i++) {
        for (int j = 0; j < NET_WIDTH; j++) {
            imageArrayNormal[0][i][j] = (imageArrayNormal[0][i][j] - mean[0]) / std[0];
            imageArrayNormal[1][i][j] = (imageArrayNormal[1][i][j] - mean[1]) / std[1];
            imageArrayNormal[2][i][j] = (imageArrayNormal[2][i][j] - mean[2]) / std[2];
        }
    }
    return APP_ERR_OK;
}

APP_ERROR ResNeSt50Classify::ArrayToTensorBase(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH],
                                               MxBase::TensorBase* tensorBase) {
    const uint32_t dataSize = CHANNEL * NET_HEIGHT * NET_WIDTH * sizeof(float);
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageArray, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {CHANNEL, NET_HEIGHT, NET_WIDTH};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR ResNeSt50Classify::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR ResNeSt50Classify::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                         std::vector<std::vector<MxBase::ClassInfo>> &clsInfos) {
    APP_ERROR ret = post_->Process(inputs, clsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR ResNeSt50Classify::Process(const std::string &imgPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    ret = Resize(imageMat, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }

    cv::Mat cropImage;
    ret = Crop(imageMat, cropImage);
    if (ret != APP_ERR_OK) {
        LogError << "Crop failed, ret=" << ret << ".";
        return ret;
    }

    float img_array[CHANNEL][NET_HEIGHT][NET_WIDTH];
    ret = Transe(cropImage, img_array);
    if (ret != APP_ERR_OK) {
        LogError << "Transe failed, ret=" << ret << ".";
        return ret;
    }

    float img_normalize[CHANNEL][NET_HEIGHT][NET_WIDTH];
    NormalizeImage(img_array, img_normalize);

    MxBase::TensorBase tensorBase;
    ret = ArrayToTensorBase(img_normalize, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "ArrayToTensorBase failed, ret=" << ret << ".";
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
    std::vector<std::vector<MxBase::ClassInfo>> BatchClsInfos = {};
    ret = PostProcess(outputs, BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    ret = SaveResult(imgPath, BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Export result to file failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR ResNeSt50Classify::SaveResult(const std::string &imgPath,
                                        const std::vector<std::vector<MxBase::ClassInfo>> &BatchClsInfos) {
    uint32_t batchIndex = 0;
    std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = fileName.find_last_of(".");
    for (const auto &clsInfos : BatchClsInfos) {
        std::string resultStr;
        for (const auto &clsInfo : clsInfos) {
            resultStr += std::to_string(clsInfo.classId) + ",";
        }
        tfile_ << fileName.substr(0, dot) << " " << resultStr << std::endl;
        if (tfile_.fail()) {
            LogError << "Failed to write the result to file.";
            return APP_ERR_COMM_WRITE_FAIL;
        }
        batchIndex++;
    }
    return APP_ERR_OK;
}


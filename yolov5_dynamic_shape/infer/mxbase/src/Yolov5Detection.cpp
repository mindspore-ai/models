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

#include "Yolov5Detection.h"
#include <unistd.h>
#include <sys/stat.h>
#include <memory>
#include <utility>
#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"

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

APP_ERROR Yolov5Detection::LoadLabels(const std::string &labelPath, std::map<int, std::string> *labelMap) {
    std::ifstream infile;
    // open label file
    infile.open(labelPath, std::ios_base::in);
    std::string s;
    // check label file validity
    if (infile.fail()) {
        LogError << "Failed to open label file: " << labelPath << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    labelMap->clear();
    // construct label map
    int count = 0;
    while (std::getline(infile, s)) {
        if (s[0] == '#') {
            continue;
        }
        size_t eraseIndex = s.find_last_not_of("\r\n\t");
        if (eraseIndex != std::string::npos) {
            s.erase(eraseIndex + 1, s.size() - eraseIndex);
        }
        labelMap->insert(std::pair<int, std::string>(count, s));
        count++;
    }
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::Init(const InitParam &initParam) {
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
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";
    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("BIASES_NUM", std::to_string(initParam.biasesNum));
    configData.SetJsonValue("BIASES", initParam.biases);
    configData.SetJsonValue("OBJECTNESS_THRESH", initParam.objectnessThresh);
    configData.SetJsonValue("IOU_THRESH", initParam.iouThresh);
    configData.SetJsonValue("SCORE_THRESH", initParam.scoreThresh);
    configData.SetJsonValue("YOLO_TYPE", std::to_string(initParam.yoloType));
    configData.SetJsonValue("MODEL_TYPE", std::to_string(initParam.modelType));
    configData.SetJsonValue("INPUT_TYPE", std::to_string(initParam.inputType));
    configData.SetJsonValue("ANCHOR_DIM", std::to_string(initParam.anchorDim));
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::Yolov5PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "Yolov5PostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    // load labels from file
    ret = LoadLabels(initParam.labelPath, &labelMap_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to load labels, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "End to Init Yolov5DetectionOpencv.";
    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    (*imageMat) = cv::imread(imgPath, cv::IMREAD_COLOR);
    imageWidth_ = (*imageMat).cols;
    imageHeight_ = (*imageMat).rows;

    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::Resize(cv::Mat *srcImageMat, cv::Mat *dstImageMat) {
    cv::resize((*srcImageMat), (*dstImageMat), cv::Size(MODEL_WIDTH, MODEL_HEIGHT));

    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::WhcToChw(const cv::Mat &srcImageMat, std::vector<float> *imgData) {
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

APP_ERROR Yolov5Detection::Focus(const cv::Mat &srcImageMat, float* data) {
    int outIdx = 0;
    int imgIdx = 0;
    int height = static_cast<int>(srcImageMat.rows);
    int width = static_cast<int>(srcImageMat.cols);
    int channel = static_cast<int>(srcImageMat.channels());
    int newHeight = height / 2;
    int newWidth = width / 2;
    int newChannel = MODEL_CHANNEL;

    std::vector<float> tmp;
    WhcToChw(srcImageMat, &tmp);

    for (int newC = 0; newC < newChannel; newC++) {
        int c = newC % channel;
        for (int newH = 0; newH < newHeight; newH++) {
            for (int newW = 0; newW < newWidth; newW++) {
                if (newC < channel) {
                    outIdx = newC * newHeight * newWidth + newH * newWidth + newW;
                    imgIdx = c * height * width + newH * 2 * width + newW * 2;
                } else if (channel <= newC && newC < channel * 2) {
                    outIdx = newC * newHeight * newWidth + newH * newWidth + newW;
                    imgIdx = c * height * width + static_cast<int>((newH + 0.5) * 2 * width) + newW * 2;
                } else if (channel * 2 <= newC && newC < channel * 3) {
                    outIdx = newC * newHeight * newWidth + newH * newWidth + newW;
                    imgIdx = c * height * width + newH * 2 * width + static_cast<int>((newW + 0.5) * 2);
                } else if (channel * 3 <= newC && newC < channel * 4) {
                    outIdx = newC * newHeight * newWidth + newH * newWidth + newW;
                    imgIdx = c * height * width + static_cast<int>((newH + 0.5) * 2 * width) +
                            static_cast<int>((newW + 0.5) * 2);
                } else {
                    LogError << "new channels Out of range.";
                    return APP_ERR_OK;
                }
                data[outIdx] = tmp[imgIdx];
            }
        }
    }

    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::CVMatToTensorBase(float* data, MxBase::TensorBase *tensorBase) {
    uint32_t height = MODEL_HEIGHT / 2;
    uint32_t width = MODEL_WIDTH / 2;
    const uint32_t dataSize = MODEL_CHANNEL * height * width * sizeof(float);
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {BATCH_NUM, MODEL_CHANNEL, height, width};
    (*tensorBase) = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::Inference(const std::vector<MxBase::TensorBase> &inputs,
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

APP_ERROR Yolov5Detection::PostProcess(const std::vector<MxBase::TensorBase>& tensors,
                                       std::vector<std::vector<MxBase::ObjectInfo>> *objInfos) {
    MxBase::ResizedImageInfo imgInfo;
    imgInfo.widthOriginal = imageWidth_;
    imgInfo.heightOriginal = imageHeight_;
    imgInfo.widthResize = MODEL_WIDTH;
    imgInfo.heightResize = MODEL_HEIGHT;
    imgInfo.resizeType = MxBase::RESIZER_STRETCHING;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);

    APP_ERROR ret = post_->Process(tensors, (*objInfos), imageInfoVec);
    if (ret != APP_ERR_OK) {
        LogInfo << "Process failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::WriteResult(const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos,
                                       const std::string &imgPath, std::vector<std::string> *jsonText) {
    uint32_t batchSize = objInfos.size();

    int pos = imgPath.rfind('/');
    std::string fileName(imgPath, pos + 1);
    fileName = fileName.substr(0, fileName.find('.'));
    // write inference result into file
    int image_id = std::stoi(fileName), cnt = 0;
    for (uint32_t i = 0; i < batchSize; i++) {
        for (auto &obj : objInfos[i]) {
            jsonText->push_back("{\"image_id\": " + std::to_string(image_id) + ", \"category_id\": " +
                                std::to_string(coco_class_nameid[static_cast<int>(obj.classId)]) + ", \"bbox\": [" +
                                std::to_string(obj.x0) + ", " + std::to_string(obj.y0) + ", " +
                                std::to_string(obj.x1 - obj.x0) + ", " + std::to_string(obj.y1 - obj.y0) + "], " +
                                "\"score\": " + std::to_string(obj.confidence) + "}");
            cnt++;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR Yolov5Detection::Process(const std::string &imgPath, std::vector<std::string> *jsonText) {
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

    float data[DATA_SIZE];
    ret = Focus(imageMat, data);

    if (ret != APP_ERR_OK) {
        LogError << "Focus failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(data, &tensorBase);

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


    std::vector<std::vector<MxBase::ObjectInfo>> objInfos;
    ret = PostProcess(outputs, &objInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    ret = WriteResult(objInfos, imgPath, jsonText);
    if (ret != APP_ERR_OK) {
        LogError << "WriteResult failed, ret=" << ret << ".";
        return ret;
    }

    imageMat.release();
    return APP_ERR_OK;
}

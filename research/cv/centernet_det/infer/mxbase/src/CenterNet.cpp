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
#include <algorithm>
#include <utility>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "CenterNet.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include <boost/property_tree/json_parser.hpp>

namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t MODEL_HEIGHT = 512;
    const uint32_t MODEL_WIDTH = 512;
}

void PrintTensorShape(const std::vector<MxBase::TensorDesc> &tensorDescVec, const std::string &tensorName) {
    LogInfo << "The shape of " << tensorName << " is as follows:";
    for (size_t i = 0; i < tensorDescVec.size(); ++i) {
        LogInfo << "  Tensor " << i << ":";
        for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
            LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
        }
    }
}

APP_ERROR CenterNet::Init(const InitParam &initParam) {
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

    PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
    PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");

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

    post_ = std::make_shared<MxBase::CenterNetMindsporePost>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "CenterNetPostProcess init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR CenterNet::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR CenterNet::ReadImage(const std::string &imgPath, cv::Mat &imageMat, ImageShape &imgShape) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    imgShape.width = imageMat.cols;
    imgShape.height = imageMat.rows;
    return APP_ERR_OK;
}

APP_ERROR CenterNet::Resize_Affine(const cv::Mat &srcImage, cv::Mat &dstImage, ImageShape &imgShape) {
    int new_width, new_height;
    new_height = static_cast<int>(imgShape.height);
    new_width = static_cast<int>(imgShape.width);
    float ss = static_cast<float>(YUV_BYTE_DE);
    cv::Mat src(new_height, new_width, CV_8UC3, srcImage.data);
    cv::Point2f srcPoint2f[3], dstPoint2f[3];
    int max_h_w = std::max(static_cast<int>(imgShape.width), static_cast<int>(imgShape.height));
    srcPoint2f[0] = cv::Point2f(static_cast<float>(new_width / ss),
                                static_cast<float>(new_height / ss));
    srcPoint2f[1] = cv::Point2f(static_cast<float>(new_width / ss),
                                static_cast<float>((new_height - max_h_w) / ss));
    srcPoint2f[2] = cv::Point2f(static_cast<float>((new_width - max_h_w) / ss),
                                static_cast<float>((new_height - max_h_w) / ss));
    dstPoint2f[0] = cv::Point2f(static_cast<float>(MODEL_WIDTH) / ss,
                                static_cast<float>(MODEL_HEIGHT) / ss);
    dstPoint2f[1] = cv::Point2f(static_cast<float>(MODEL_WIDTH) / ss, 0.0);
    dstPoint2f[2] = cv::Point2f(0.0, 0.0);

    cv::Mat warp_mat(2, 3, CV_32FC1);
    warp_mat = cv::getAffineTransform(srcPoint2f, dstPoint2f);
    cv::Mat warp_dst = cv::Mat::zeros(cv::Size(static_cast<int>(MODEL_HEIGHT), static_cast<int>(MODEL_WIDTH)),
                                      src.type());
    cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());
    cv::Mat dst;
    warp_dst.convertTo(dst, CV_32F);
    dstImage = dst;
    // nomalization
    float mean[3] = {0.40789654, 0.44719302, 0.47026115};
    float std[3] = {0.28863828, 0.27408164, 0.27809835};
    std::vector<cv::Mat> channels;
    cv::split(dstImage, channels);
    cv::Mat blue, green, red;
    blue = channels.at(0);
    green = channels.at(1);
    red = channels.at(2);
    cv::Mat B, G, R;
    float mm = 255;
    B = ((blue/mm) - mean[0]) / std[0];
    G = ((green/mm) - mean[1]) / std[1];
    R = ((red/mm) - mean[2]) / std[2];
    std::vector<cv::Mat> channels2;
    channels2.push_back(B);
    channels2.push_back(G);
    channels2.push_back(R);
    cv::merge(channels2, dstImage);

    return APP_ERR_OK;
}

APP_ERROR CenterNet::CVMatToTensorBase(std::vector<float> &imageData, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize = MODEL_HEIGHT * MODEL_WIDTH * MxBase::YUV444_RGB_WIDTH_NU * YUV_BYTE_DE * YUV_BYTE_DE;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(&imageData[0]),
                                     dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {MODEL_HEIGHT * MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(MODEL_WIDTH)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR CenterNet::Inference(const std::vector<MxBase::TensorBase> &inputs,
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

APP_ERROR CenterNet::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
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

void SaveInferResult(const std::vector<MxBase::ObjectInfo> &objInfos, const std::string &resultPath) {
    if (objInfos.empty()) {
        LogWarn << "The predict result is empty.";
        return;
    }

    namespace pt = boost::property_tree;
    pt::ptree root, data;
    int index = 0;
    for (auto &obj : objInfos) {
        ++index;
        LogInfo << "BBox[" << index << "]:[x0=" << obj.x0 << ", y0=" << obj.y0 << ", x1=" << obj.x1 << ", y1=" << obj.y1
                << "], confidence=" << obj.confidence << ", classId=" << obj.classId << ", className=" << obj.className
                << std::endl;
        pt::ptree item;
        item.put("classId", obj.classId);
        item.put("className", obj.className);
        item.put("confidence", obj.confidence);
        item.put("x0", obj.x0);
        item.put("y0", obj.y0);
        item.put("x1", obj.x1);
        item.put("y1", obj.y1);

        data.push_back(std::make_pair("", item));
    }
    root.add_child("data", data);
    pt::json_parser::write_json(resultPath, root, std::locale(), true);
}

APP_ERROR CenterNet::Process(const std::string &imgPath, const std::string &resultPath) {
    cv::Mat imageMat;
    ImageShape imageShape{};
    APP_ERROR ret = ReadImage(imgPath, imageMat, imageShape);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    cv::Mat dstImage(MODEL_WIDTH, MODEL_HEIGHT, CV_32FC3);
    Resize_Affine(imageMat, dstImage, imageShape);

    std::vector<float> dst_data;
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(dstImage, bgrChannels);
    for (std::size_t i = 0; i < bgrChannels.size(); i++) {
        std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
        dst_data.insert(dst_data.end(), data.begin(), data.end());
    }

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(dst_data, tensorBase);
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
    std::vector<std::vector<MxBase::ObjectInfo>> objectInfos = {};
    std::vector<MxBase::ResizedImageInfo> resizedImageInfos = {};
    MxBase::ResizedImageInfo imgInfo = {
      MODEL_WIDTH, MODEL_HEIGHT, imageShape.width, imageShape.height, MxBase::RESIZER_STRETCHING, 0.0};
    resizedImageInfos.push_back(imgInfo);
    std::map<std::string, std::shared_ptr<void>> configParamMap = {};
    ret = PostProcess(outputs, objectInfos, resizedImageInfos, configParamMap);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::ObjectInfo> objects = objectInfos.at(0);
    SaveInferResult(objects, resultPath);
    return APP_ERR_OK;
}

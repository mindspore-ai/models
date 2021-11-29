/*
 * Copyright 2021 Huawei Technologies Co., Ltd.
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
#include <map>
#include "InceptionResnetV2Classify.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR InceptionResnetV2Classify::Init(const InitParam &initParam) {
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
    return APP_ERR_OK;
}

APP_ERROR InceptionResnetV2Classify::DeInit() {
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

void InceptionResnetV2Classify::ReadImage(const std::string &imgPath,
    cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
}

void InceptionResnetV2Classify::CenterCropImage(const cv::Mat &img, cv::Mat &cropImg) {
    float central_fraction = 0.86;
    int crop_x = img.cols * central_fraction + 1;
    int crop_y = img.rows * central_fraction + 1;
    int crop_x1 = (img.cols - crop_x) / 2;
    int crop_y1 = (img.rows - crop_y) / 2;

    cv::Rect myROI(crop_x1, crop_y1, crop_x, crop_y);
    LogInfo << "images crop_x1: " <<  crop_x1 << ", crop_x: " << crop_x <<
    ", crop_y1: " << crop_y1 << ", crop_y: " << crop_y;
    cropImg = img(myROI);
}

void InceptionResnetV2Classify::Resize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    static constexpr uint32_t resizeHeight = 299;
    static constexpr uint32_t resizeWidth = 299;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
}

APP_ERROR InceptionResnetV2Classify::CVMatToTensorBase(const cv::Mat &imageMat,
    MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);

    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR InceptionResnetV2Classify::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
    auto startTime = std::chrono::high_resolution_clock::now();  // search for learning
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

APP_ERROR InceptionResnetV2Classify::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
    std::vector<std::vector<MxBase::ClassInfo>> &clsInfos) {
    APP_ERROR ret = post_->Process(inputs, clsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR InceptionResnetV2Classify::SaveInferResult(const std::string &imagePath,
    const std::vector<std::vector<MxBase::ClassInfo>> &batchClsInfos) {
    uint32_t batchIndex = 0;
    LogInfo << "image path: " << imagePath;
    std::string fileName = imagePath.substr(imagePath.find_last_of("/") + 1);
    size_t dot = fileName.find_last_of(".");
    std::string resultPathName = "result";
    if (access(resultPathName.c_str(), 0) != 0) {
        APP_ERROR ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << ret;
            return APP_ERR_COMM_FAILURE;
        }
    }

    std::string resFileName = "result/" + fileName.substr(0, dot) + "_1.txt";

    LogInfo << "file path for saving result: " << resFileName;
    std::ofstream tfile(resFileName);
    if (tfile.fail()) {
        LogError << "Failed to open result file";
        return APP_ERR_COMM_FAILURE;
    }
    for (auto clsInfos : batchClsInfos) {
        std::string resultStr = "";
        for (auto clsInfo : clsInfos) {
            LogDebug << "batchIndex: " << batchIndex << " className: " << clsInfo.className
                     << " confidence: " << clsInfo.confidence << " classIndex: " << clsInfo.classId;
            resultStr += std::to_string(clsInfo.classId) + " ";
        }
        tfile << resultStr << std::endl;
        batchIndex += 1;
    }
    tfile.close();
    return APP_ERR_OK;
}

APP_ERROR InceptionResnetV2Classify::Process(const std::string &imgPath) {
    cv::Mat imageMat;
    ReadImage(imgPath, imageMat);
    CenterCropImage(imageMat, imageMat);
    Resize(imageMat, imageMat);

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    APP_ERROR ret = CVMatToTensorBase(imageMat, tensorBase);
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
    std::vector<std::vector<MxBase::ClassInfo>> BatchClsInfos = {};
    ret = PostProcess(outputs, BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    ret = SaveInferResult(imgPath, BatchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Save results failed, ret: " << ret << ".";
        return ret;
    }
    imageMat.release();
    return APP_ERR_OK;
}

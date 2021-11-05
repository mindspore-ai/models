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
#include <dirent.h>
#include <memory>
#include <map>
#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include "FaceRecognitionForTracking.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"


APP_ERROR FaceRecognitionForTracking::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    // 设备初始化
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }

    // 上下文初始化
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }

    // 加载模型
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::ReadImage(const std::string &imgPath, cv::Mat &imgMat) {
    imgMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(imgMat, imgMat, cv::COLOR_BGR2RGB);
    if (imgMat.empty()) {
        LogError << "imread failed. img: " << imgPath;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::Resize(const cv::Mat &srcMat, float *dstMat) {
    static constexpr uint32_t resizeWidth = 64;
    static constexpr uint32_t resizeHeight = 96;
    cv::Mat resizeMat;
    cv::resize(srcMat, resizeMat, cv::Size(resizeWidth, resizeHeight), 0, 0, cv::INTER_LINEAR);
    for (auto i = 0; i < resizeMat.rows; i++) {
        for (auto j = 0; j < resizeMat.cols; j++) {
            dstMat[i * resizeWidth + j] = (static_cast<float>(resizeMat.at<cv::Vec3b>(i, j)[0]) / 255 - 0.5) * 2;
            dstMat[resizeHeight * resizeWidth + i * resizeWidth + j] =\
(static_cast<float>(resizeMat.at<cv::Vec3b>(i, j)[1]) / 255 - 0.5) * 2;
            dstMat[2 * resizeHeight * resizeWidth + i * resizeWidth + j] =\
(static_cast<float>(resizeMat.at<cv::Vec3b>(i, j)[2]) / 255 - 0.5) * 2;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::CvMatToTensorBase(float* imgMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize = 3*96*64*sizeof(float);

    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imgMat, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {3, static_cast<uint32_t>(96*64)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);

    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::Inference(const std::vector<MxBase::TensorBase> &inputs, \
    std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); i++) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); j++) {
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
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMs += costMs;
    return APP_ERR_OK;
}

void FaceRecognitionForTracking::InclassLikehood(cv::Mat &featureMat, std::vector<std::string> &names, \
    std::vector<float> &inclassLikehood) {
    std::map<std::string, cv::Mat> objFeatures;
    for (uint32_t i = 0; i < names.size(); i++) {
        std::string objName = names[i].substr(0, names[i].size() - 5);
        objFeatures[objName].push_back(featureMat.row(i));
    }
    for (auto it : objFeatures) {
        cv::Mat objFeature = it.second;
        cv::Mat objFeatureT = objFeature.t();
        cv::Mat mul = objFeature * objFeatureT;
        for (auto i = 0; i < mul.rows; i++) {
            for (auto j = 0; j < mul.cols; j++) {
                inclassLikehood.push_back(mul.at<float>(i, j));
            }
        }
    }
}

void FaceRecognitionForTracking::BtclassLikehood(cv::Mat &featureMat, std::vector<std::string> &names, \
    std::vector<float> &bnclassLikehood) {
    for (uint32_t i = 0; i < names.size(); i++) {
        std::string objName = names[i].substr(0, names[i].size() - 5);
        for (uint32_t j = 0; j < names.size(); j++) {
            std::string objName2 = names[j].substr(0, names[j].size() - 5);
            if (objName == objName2) continue;
            cv::Mat feature = featureMat.row(i);
            bnclassLikehood.push_back(feature.dot(featureMat.row(j)));
        }
    }
}

void FaceRecognitionForTracking::TarAtFar(std::vector<float> &inclassLikehood, std::vector<float> &btclassLikehood,
                                            std::vector<std::vector<float>> &tarFars) {
    std::vector<float> points = {0.5, 0.3, 0.1, 0.01, 0.001, 0.0001, 0.00001};
    for (auto point : points) {
        float thre = btclassLikehood[static_cast<int>(btclassLikehood.size() * point)];
        int index = std::upper_bound(inclassLikehood.begin(), inclassLikehood.end(), thre) - inclassLikehood.begin();
        int n = inclassLikehood.size() - index - 1;
        std::vector<float> tarFar;
        tarFar.push_back(point);
        tarFar.push_back(1.0 * n / inclassLikehood.size());
        tarFar.push_back(thre);
        tarFars.push_back(tarFar);
    }
}

APP_ERROR FaceRecognitionForTracking::PostProcess(const std::vector<MxBase::TensorBase> &inputs, \
    std::vector<std::string> &names) {
    auto featureShape = inputs[0].GetShape();
    cv::Mat featureMat(inputs.size(), featureShape[1], CV_32FC1);
    for (uint32_t i = 0; i < inputs.size(); i++) {
        MxBase::TensorBase feature = inputs[i];
        feature.ToHost();
        auto data = reinterpret_cast<float(*)>(feature.GetBuffer());
        for (uint32_t j = 0; j < featureShape[1]; j++) {
            featureMat.at<float>(i, j) = data[j];
        }
    }

    std::vector<float> inclassLikehood;
    InclassLikehood(featureMat, names, inclassLikehood);
    std::sort(inclassLikehood.begin(), inclassLikehood.end());

    std::vector<float> btclassLikehood;
    BtclassLikehood(featureMat, names, btclassLikehood);
    std::sort(btclassLikehood.begin(), btclassLikehood.end(), std::greater<float>());

    std::vector<std::vector<float>> tarFars;
    TarAtFar(inclassLikehood, btclassLikehood, tarFars);

    for (uint32_t i = 0; i < tarFars.size(); i++) {
        std::cout << "---" << tarFars[i][0] << ": " << tarFars[i][1] << "@" << tarFars[i][2] << std::endl;
    }

    return APP_ERR_OK;
}

std::vector<std::string> FaceRecognitionForTracking::GetFileList(const std::string &dirPath) {
    struct dirent *ptr;
    DIR *dir = opendir(dirPath.c_str());
    std::vector<std::string> files;
    while ((ptr = readdir(dir)) != NULL) {
        if (ptr->d_name[0] == '.') continue;
        files.push_back(ptr->d_name);
    }
    closedir(dir);
    return files;
}

APP_ERROR FaceRecognitionForTracking::Process(const std::string &dirPath) {
    std::vector<std::string> dirFileList = GetFileList(dirPath);
    std::vector<std::string> names, paths;
    for (auto fileList : dirFileList) {
        std::string subPaht = dirPath + "/" + fileList;
        std::vector<std::string> files = GetFileList(subPaht);
        for (auto imgFile : files) {
            std::string name = imgFile.substr(0, imgFile.find("."));
            std::string path = subPaht + "/" + imgFile;
            names.push_back(name);
            paths.push_back(path);
        }
    }

    std::vector<MxBase::TensorBase> features;
    for (auto imgPath : paths) {
        cv::Mat image;
        APP_ERROR ret = ReadImage(imgPath, image);
        if (ret != APP_ERR_OK) {
            LogError << "ReadImage failed, ret=" << ret << ".";
            return ret;
        }
        float resizeImage[18432];
        ret = Resize(image, resizeImage);
        if (ret != APP_ERR_OK) {
            LogError << "Resize failed, ret=" << ret << ".";
            return ret;
        }
        MxBase::TensorBase imageTensor;
        ret = CvMatToTensorBase(resizeImage, imageTensor);
        if (ret != APP_ERR_OK) {
            LogError << "CvMatToTensorBase failed, ret=" << ret << ".";
            return ret;
        }
        std::vector<MxBase::TensorBase> inputs = {};
        std::vector<MxBase::TensorBase> outputs = {};
        inputs.push_back(imageTensor);
        ret = Inference(inputs, outputs);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        features.push_back(outputs[0]);
    }

    APP_ERROR ret = PostProcess(features, names);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

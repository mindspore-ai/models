/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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


#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <memory>
#include <vector>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "FaceAttribute.h"

namespace {
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
    const int attributeNum[3] = {9, 2, 2};
}

APP_ERROR faceattribute::Init(const InitParam &initParam) {
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
    std::map<std::string, std::shared_ptr<void> > config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::Resnet50PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "faceattribute init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR faceattribute::DeInit() {
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


cv::Mat faceattribute::BGRToRGB(cv::Mat img) {
        cv::Mat image(img.rows, img.cols, CV_8UC3);
        for (int i = 0; i < img.rows; ++i) {
                // Get the first pixel pointer of row i
                cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
                cv::Vec3b *p2 = image.ptr<cv::Vec3b>(i);
                for (int j = 0; j < img.cols; ++j) {
                        // Convert bgr of img to rgb of image
                        p2[j][2] = p1[j][0];
                        p2[j][1] = p1[j][1];
                        p2[j][0] = p1[j][2];
                }
        }
        return image;
}


bool faceattribute::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = BGRToRGB(cv::imread(imgPath, cv::IMREAD_COLOR));
    if (!imageMat.empty()) {
        return true;
    }
    return false;
}

APP_ERROR faceattribute::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    static constexpr uint32_t resizeHeight = 112;
    static constexpr uint32_t resizeWidth = 112;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR faceattribute::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize =  imageMat.cols *  imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;
    LogInfo << "image size is " << imageMat.cols << " " << imageMat.rows;
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

APP_ERROR faceattribute::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR faceattribute::PostProcess(std::vector<MxBase::TensorBase> &tensors, int classNum[]) {
    LogDebug << "Start to PostProcess";

    APP_ERROR ret = APP_ERR_OK;
    for (MxBase::TensorBase &input : tensors) {
        ret = input.ToHost();
        if (ret != APP_ERR_OK) {
            LogError << "----------Error occur!!";
        }
    }
    auto inputs = tensors;

    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed. ret=" << ret;
        return ret;
    }
    // View the length of the array
    int length = 3;
    // You know how many attributes there are by their length
    for (uint32_t i = 0; i < length; i++) {
            const uint32_t softmaxTensorIndex = i;
            auto softmaxTensor = inputs[softmaxTensorIndex];
            auto shape = softmaxTensor.GetShape();
            uint32_t batchSize = shape[0];
            void *softmaxTensorPtr = softmaxTensor.GetBuffer();
            int index;
            for (int istart = 0; istart < batchSize; istart++) {
                float max = FLT_MIN;
                index = 0;
                for (int j = 0; j < attributeNum[i]; j++) {
                    float value = *(static_cast<float *>(softmaxTensorPtr) + istart * attributeNum[i] + j);
                    if (max < value) {
                        max = value;
                        index = j;
                    }
               }
           }
            // get the max value and its index
            classNum[i] = index;
        }
    LogDebug << "End to PostProcess";
    return APP_ERR_OK;
}

APP_ERROR faceattribute::Compare(int ground_class[], int class_predict[] ) {
    // int length = sizeof (class_predict) / sizeof (class_predict[0]);
    int length = 3;
    // Compare the results. If equal, assign the value of ground_class to 1, otherwise assign it to 0.
    for (int i = 0; i < length; i++) {
        if (ground_class[i] == class_predict[i]) {
            ground_class[i] = 1;
        } else {
            ground_class[i] = 0;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR faceattribute::Process(const std::string &imgPath, int ground_class[]) {
    cv::Mat imageMat;
    bool readTrue = ReadImage(imgPath, imageMat);
    if (!readTrue) {
        LogError << "ReadImage failed!!!" << std::endl;
    }
    // resize image
    ResizeImage(imageMat, imageMat);
    // Create input and output data types to be stored
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    // Convert input data to tensor format
    MxBase::TensorBase tensorBase;
    APP_ERROR ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);

    auto startTime = std::chrono::high_resolution_clock::now();

    // start to inference
    ret = Inference(inputs, outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    int class_predict[3];
    // Post-processing of prediction results
    ret = PostProcess(outputs, class_predict);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    // Comparison with real labels
    ret = Compare(ground_class, class_predict);
    if (ret != APP_ERR_OK) {
        LogError << "Compare fail !!!" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}



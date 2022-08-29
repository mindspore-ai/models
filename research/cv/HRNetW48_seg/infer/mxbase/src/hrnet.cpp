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

#include "src/hrnet.h"

#include <memory>
#include <vector>
#include <string>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"

namespace localParameter {
    const uint32_t VECTOR_FIRST_INDEX = 0;
    const uint32_t VECTOR_SECOND_INDEX = 1;
    const uint32_t VECTOR_THIRD_INDEX = 2;
    const uint32_t VECTOR_FOURTH_INDEX = 3;
    const uint32_t VECTOR_FIFTH_INDEX = 4;
}

APP_ERROR HRNetW48Seg::Init(const InitParam& initParam) {
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

    return APP_ERR_OK;
}

APP_ERROR HRNetW48Seg::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();

    return APP_ERR_OK;
}

APP_ERROR HRNetW48Seg::ReadImage(const std::string& imgPath, cv::Mat& imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);

    return APP_ERR_OK;
}

APP_ERROR HRNetW48Seg::CVMatToTensorBase(const cv::Mat& imageMat, MxBase::TensorBase& tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = { imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols) };
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);

    return APP_ERR_OK;
}

APP_ERROR HRNetW48Seg::Inference(const std::vector<MxBase::TensorBase>& inputs,
    std::vector<MxBase::TensorBase>& outputs) {
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
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR HRNetW48Seg::Process(const std::string& imgPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::ResizedImageInfo resizedImageInfo;
    resizedImageInfo.heightOriginal = imageMat.rows;
    resizedImageInfo.widthOriginal = imageMat.cols;
    resizedImageInfo.resizeType = MxBase::RESIZER_STRETCHING;
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
    cv::Mat output;
    ret = PostProcess(outputs, resizedImageInfo, output);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    std::string resultPath = imgPath;
    size_t pos = resultPath.find_last_of(".");
    resultPath.replace(resultPath.begin() + pos, resultPath.end(), "_infer.png");
    SaveResultToImage(output, resultPath);

    return APP_ERR_OK;
}

APP_ERROR HRNetW48Seg::PostProcess(std::vector<MxBase::TensorBase>& inputs,
    const MxBase::ResizedImageInfo& resizedInfo, cv::Mat& output) {
    MxBase::TensorBase& tensor = inputs[0];
    int ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    uint32_t imgHeight = resizedInfo.heightOriginal;
    uint32_t imgWidth = resizedInfo.widthOriginal;
    uint32_t outputModelWidth = tensor.GetShape()[localParameter::VECTOR_FOURTH_INDEX];
    uint32_t outputModelHeight = tensor.GetShape()[localParameter::VECTOR_THIRD_INDEX];
    uint32_t outputModelChannel = tensor.GetShape()[localParameter::VECTOR_SECOND_INDEX];
    auto data = reinterpret_cast<float(*)[outputModelHeight][outputModelWidth]>(tensor.GetBuffer());
    std::vector<cv::Mat> tensorChannels = {};
    for (size_t c = 0; c < outputModelChannel; ++c) {
        cv::Mat channelMat(outputModelHeight, outputModelWidth, CV_32FC1);
        for (size_t h = 0; h < outputModelHeight; ++h) {
            for (size_t w = 0; w < outputModelWidth; ++w) {
                channelMat.at<float>(h, w) = data[c][h][w];
            }
        }
        tensorChannels.push_back(channelMat);
    }
    cv::Mat argmax(imgHeight, imgWidth, CV_8UC1);
    for (size_t h = 0; h < imgHeight; ++h) {
        for (size_t w = 0; w < imgWidth; ++w) {
            size_t cMax = 0;
            for (size_t c = 0; c < outputModelChannel; ++c) {
                if (tensorChannels[c].at<float>(h, w) > tensorChannels[cMax].at<float>(h, w)) {
                    cMax = c;
                }
            }
            argmax.at<uchar>(h, w) = cMax;
        }
    }
    output = argmax;

    return APP_ERR_OK;
}

void HRNetW48Seg::SaveResultToImage(const cv::Mat& imageMat, const std::string& filePath) {
    std::cout << "Save result to image: " << filePath << std::endl;
    cv::Mat imageGrayC3 = cv::Mat::zeros(imageMat.rows, imageMat.cols, CV_8UC3);
    std::vector<cv::Mat> planes;
    for (int i = 0; i < 3; i++) {
        planes.push_back(imageMat);
    }
    cv::merge(planes, imageGrayC3);
    uchar rgbColorMap[256 * 3] = {
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
        70, 130, 180,
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

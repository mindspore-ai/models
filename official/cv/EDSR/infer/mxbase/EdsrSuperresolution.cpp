/**
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
#include "EdsrSuperresolution.h"

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

APP_ERROR EdsrSuperresolution::Init(const InitParam &initParam) {
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
    uint32_t outputModelHeight = modelDesc_.outputTensors[0].tensorDims[localParameter::VECTOR_THIRD_INDEX];
    uint32_t inputModelHeight = modelDesc_.inputTensors[0].tensorDims[localParameter::VECTOR_SECOND_INDEX];
    uint32_t inputModelWidth = modelDesc_.inputTensors[0].tensorDims[localParameter::VECTOR_THIRD_INDEX];

    scale_ = outputModelHeight/inputModelHeight;
    maxEdge_ = inputModelWidth > inputModelHeight ? inputModelWidth:inputModelHeight;
    return APP_ERR_OK;
}

APP_ERROR EdsrSuperresolution::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR EdsrSuperresolution::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    imageWidth_ = imageMat->cols;
    imageHeight_ = imageMat->rows;
    return APP_ERR_OK;
}

APP_ERROR EdsrSuperresolution::PaddingImage(cv::Mat *imageSrc, cv::Mat *imageDst, const uint32_t &targetLength) {
    uint32_t padding_h = targetLength - imageHeight_;
    uint32_t padding_w = targetLength - imageWidth_;
    cv::copyMakeBorder(*imageSrc, *imageDst, 0, padding_h, 0, padding_w, cv::BORDER_CONSTANT, 0);
    return APP_ERR_OK;
}


APP_ERROR EdsrSuperresolution::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;

    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);

    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR EdsrSuperresolution::Inference(std::vector<MxBase::TensorBase> *inputs,
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
        outputs->push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(*inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR EdsrSuperresolution::PostProcess(std::vector<MxBase::TensorBase> *inputs, cv::Mat *imageMat) {
        MxBase::TensorBase tensor = *inputs->begin();
        int ret = tensor.ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Tensor deploy to host failed.";
            return ret;
        }
        uint32_t outputModelChannel = tensor.GetShape()[localParameter::VECTOR_SECOND_INDEX];
        uint32_t outputModelHeight = tensor.GetShape()[localParameter::VECTOR_THIRD_INDEX];
        uint32_t outputModelWidth = tensor.GetShape()[localParameter::VECTOR_FOURTH_INDEX];
        LogInfo << "Channel:" << outputModelChannel << " Height:" << outputModelHeight << " Width:" << outputModelWidth;

        uint32_t finalHeight = imageHeight_ * scale_;
        uint32_t finalWidth = imageWidth_ * scale_;
        cv::Mat output(finalHeight, finalWidth, CV_32FC3);

        auto data = reinterpret_cast<float(*)[outputModelChannel]
        [outputModelHeight][outputModelWidth]>(tensor.GetBuffer());

        for (size_t c = 0; c < outputModelChannel; ++c) {
            for (size_t x = 0; x < finalHeight; ++x) {
                for (size_t y = 0; y < finalWidth; ++y) {
                    output.at<cv::Vec3f>(x, y)[c] = data[0][c][x][y];
                }
            }
        }

        *imageMat = output;
        return APP_ERR_OK;
}

APP_ERROR EdsrSuperresolution::Process(const std::string &imgPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, &imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    PaddingImage(&imageMat, &imageMat, maxEdge_);
    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }


    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(tensorBase);
    ret = Inference(&inputs, &outputs);

    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    cv::Mat output;
    ret = PostProcess(&outputs, &output);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    std::string resultPath = imgPath;
    size_t pos = resultPath.find_last_of(".");
    resultPath.replace(resultPath.begin() + pos, resultPath.end(), "_infer.png");
    cv::imwrite(resultPath, output);
    return APP_ERR_OK;
}

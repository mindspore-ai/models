/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "red30Superresolution.h"

#include <memory>
#include <vector>
#include <string>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"


APP_ERROR red30Superresolution::Init(const InitParam &initParam) {
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

    uint32_t inputModelHeight = modelDesc_.inputTensors[0].tensorDims[MxBase::VECTOR_SECOND_INDEX];
    uint32_t inputModelWidth = modelDesc_.inputTensors[0].tensorDims[MxBase::VECTOR_THIRD_INDEX];

    TargetHeight_ = inputModelHeight;
    TargetWidth_ = inputModelWidth;
    LogInfo << " TargetHeight_:" << TargetHeight_ << " TargetWidth_:" <<TargetWidth_;
    return APP_ERR_OK;
}
APP_ERROR red30Superresolution::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR red30Superresolution::PaddingImage(cv::Mat *imageSrc, cv::Mat *imageDst,
                                             const uint32_t &target_h, const uint32_t &target_w) {
    uint32_t padding_h = target_h - imageHeight_;
    uint32_t padding_w = target_w - imageWidth_;
    cv::copyMakeBorder(*imageSrc, *imageDst, 0, padding_h, 0, padding_w, cv::BORDER_CONSTANT, 0);
    return APP_ERR_OK;
}

APP_ERROR red30Superresolution::UnPaddingImage(cv::Mat *imageSrc, const uint32_t &target_h, const uint32_t &target_w) {
    cv::Rect m_select(0, 0, target_w, target_h);
    *imageSrc = (*imageSrc)(m_select);
    return APP_ERR_OK;
}

APP_ERROR red30Superresolution::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    imageWidth_ = imageMat->cols;
    imageHeight_ = imageMat->rows;
    return APP_ERR_OK;
}


APP_ERROR red30Superresolution::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
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

APP_ERROR red30Superresolution::Inference(std::vector<MxBase::TensorBase> *inputs,
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


APP_ERROR red30Superresolution::PostProcess(std::vector<MxBase::TensorBase> *inputs, cv::Mat *imageMat) {
        MxBase::TensorBase tensor = *inputs->begin();
        int ret = tensor.ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Tensor deploy to host failed.";
            return ret;
        }
        uint32_t outputModelChannel = tensor.GetShape()[MxBase::VECTOR_SECOND_INDEX];
        uint32_t outputModelHeight = tensor.GetShape()[MxBase::VECTOR_THIRD_INDEX];
        uint32_t outputModelWidth = tensor.GetShape()[MxBase::VECTOR_FOURTH_INDEX];
        LogInfo << "Channel:" << outputModelChannel << " Height:"
                << outputModelHeight << " Width:" <<outputModelWidth;

        cv::Mat output(outputModelHeight, outputModelWidth, CV_32FC3);

        auto data = reinterpret_cast<float(*)[outputModelChannel]
        [outputModelHeight][outputModelWidth]>(tensor.GetBuffer());

        for (size_t c = 0; c < outputModelChannel; ++c) {
            for (size_t x = 0; x < outputModelHeight; ++x) {
                for (size_t y = 0; y < outputModelWidth; ++y) {
                    output.at<cv::Vec3f>(x, y)[c] = data[0][c][x][y];
                }
            }
        }

        *imageMat = output;
        return APP_ERR_OK;
}

APP_ERROR red30Superresolution::Process(const std::string &imgPath, const std::string &resultPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, &imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    PaddingImage(&imageMat, &imageMat, TargetHeight_, TargetWidth_);
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
    UnPaddingImage(&output, imageHeight_, imageWidth_);
    std::string imgName = imgPath;
    size_t pos_begin = imgName.find_last_of("/");
    if (static_cast<int>(pos_begin) == -1) {
        imgName = "./" + imgName;
        pos_begin = imgName.find_last_of("/");
    }
    imgName.replace(imgName.begin(), imgName.begin()+pos_begin, "");
    size_t pos_end = imgName.find_last_of(".");
    imgName.replace(imgName.begin() + pos_end, imgName.end(), ".jpg");
    std::string resultPathfile = resultPath + imgName;
    LogInfo << "resultPathfile: " << resultPathfile;
    cv::imwrite(resultPathfile, output);
    return APP_ERR_OK;
}

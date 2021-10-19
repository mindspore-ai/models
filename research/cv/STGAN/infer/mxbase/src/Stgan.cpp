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

#include "Stgan.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t FLOAT32_TYPE_BYTE_NUM = 4;
    const float NORMALIZE_MEAN = 255/2;
    const float NORMALIZE_STD = 255/2;
    const uint32_t OUTPUT_HEIGHT = 128;
    const uint32_t OUTPUT_WIDTH = 128;
    const uint32_t CHANNEL = 3;
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

APP_ERROR Stgan::Init(const InitParam &initParam) {
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
    savePath_ = initParam.savePath;
    PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
    PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");

    return APP_ERR_OK;
}

APP_ERROR Stgan::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Stgan::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR Stgan::ResizeImage(const cv::Mat &srcImageMat, cv::Mat *dstImageMat) {
    static constexpr uint32_t resizeHeight = OUTPUT_HEIGHT;
    static constexpr uint32_t resizeWidth = OUTPUT_WIDTH;

    cv::resize(srcImageMat, *dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR Stgan::CVMatToTensorBase(const cv::Mat& imageMat, MxBase::TensorBase *tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }
    // mat NHWC to NCHW, BGR to RGB, and Normalize
    size_t  H = OUTPUT_HEIGHT, W = OUTPUT_WIDTH, C = CHANNEL;

    float* mat_data = new float[dataSize];
    dataSize = dataSize * FLOAT32_TYPE_BYTE_NUM;
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                int id = (C - c - 1) * (H * W) + h * W + w;
                mat_data[id] = (imageMat.at<cv::Vec3b>(h, w)[c] - NORMALIZE_MEAN) / NORMALIZE_STD;
            }
        }
    }

    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(&mat_data[0]),
                                        dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = { 1, CHANNEL, OUTPUT_HEIGHT, OUTPUT_WIDTH};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR Stgan::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
    dynamicInfo.batchSize = 1;

    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Stgan::PostProcess(std::vector<MxBase::TensorBase> outputs, cv::Mat *resultImg) {
    APP_ERROR ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
    }
    float *outputPtr = reinterpret_cast<float *>(outputs[0].GetBuffer());

    size_t  H = OUTPUT_HEIGHT, W = OUTPUT_WIDTH, C = CHANNEL;

    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                float tmpNum = *(outputPtr + (C - c - 1) * (H * W) + h * W + w) * NORMALIZE_STD + NORMALIZE_MEAN;
                resultImg->at<cv::Vec3b>(h, w)[c] = static_cast<int>(tmpNum);
            }
        }
    }

    return APP_ERR_OK;
}

APP_ERROR Stgan::SaveResult(const cv::Mat &resultImg, const std::string &imgName) {
    DIR *dirPtr = opendir(savePath_.c_str());
    if (dirPtr == nullptr) {
        std::string path1 = "mkdir -p " + savePath_;
        system(path1.c_str());
    }
    cv::imwrite(savePath_ + "/" + imgName, resultImg);
    return APP_ERR_OK;
}

APP_ERROR Stgan::GetImageLabel(std::vector<float> label, MxBase::TensorBase *imgLabels) {
    float* mat_data = new float[label.size()];
    for (size_t i = 0; i < label.size(); i++) {
        mat_data[i] = label[i];
    }
    MxBase::MemoryData memoryDataDst(label.size()*FLOAT32_TYPE_BYTE_NUM, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(&mat_data[0]),
                                        label.size()*FLOAT32_TYPE_BYTE_NUM, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    const std::vector<uint32_t> shape = {1, (unsigned int)label.size()};
    *imgLabels = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR Stgan::Process(const std::string &imgPath, const std::string &imgName, const std::vector<float> &label) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, &imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    ResizeImage(imageMat, &imageMat);

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);

    MxBase::TensorBase imgLabels;
    ret = GetImageLabel(label, &imgLabels);
    if (ret != APP_ERR_OK) {
        LogError << "Get Image label failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(imgLabels);
    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    cv::Mat resultImg(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC3);
    ret = PostProcess(outputs, &resultImg);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    ret = SaveResult(resultImg, imgName);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd.
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
#include <fstream>
#include <map>
#include "PVNet.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR PVNet::Init(const InitParam &initParam) {
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

APP_ERROR PVNet::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR PVNet::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(imageMat, imageMat, cv::COLOR_RGB2BGR);
    return APP_ERR_OK;
}

APP_ERROR PVNet::ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    static constexpr uint32_t resizeHeight = 480;
    static constexpr uint32_t resizeWidth  = 640;

    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
    return APP_ERR_OK;
}

APP_ERROR PVNet::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize =  imageMat.cols * imageMat.rows * 3;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, static_cast<uint32_t>(imageMat.rows), static_cast<uint32_t>(imageMat.cols), 3};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}



APP_ERROR PVNet::Inference(const std::vector<MxBase::TensorBase> &inputs,
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

APP_ERROR PVNet::SaveResult(const std::string &imgPath, std::vector<MxBase::TensorBase> &outputs) {
    // LogInfo << "result path" << imgPath;
    std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
    size_t dot = fileName.find_last_of(".");
    std::string segFileName = "result/seg_pred/" + fileName.substr(0, dot) + ".bin";
    std::string verFileName = "result/ver_pred/" + fileName.substr(0, dot) + ".bin";
    LogInfo << "file path for saving seg_pred:" << segFileName;
    LogInfo << "file path for saving ver_pred:" << verFileName;

    // save seg_pred
    APP_ERROR ret0 = outputs[0].ToHost();
    if (ret0 != APP_ERR_OK) {
        LogError << GetError(ret0) << "tohost fail.";
        return ret0;
    }
    void *segOutput = outputs[0].GetBuffer();
    std::vector<uint32_t> segShape = outputs[0].GetShape();
    FILE *segOutputFile = fopen(segFileName.c_str(), "wb");
    fwrite(segOutput, segShape[0]*segShape[1]*segShape[2]*segShape[3], sizeof(float), segOutputFile);
    fclose(segOutputFile);
    segOutputFile = nullptr;

    // save ver_pred
    APP_ERROR ret1 = outputs[1].ToHost();
    if (ret1 != APP_ERR_OK) {
        LogError << GetError(ret1) << "tohost fail.";
        return ret1;
    }
    void *verOutput = outputs[1].GetBuffer();
    std::vector<uint32_t> verShape = outputs[1].GetShape();
    FILE *verOutputFile = fopen(verFileName.c_str(), "wb");
    fwrite(verOutput, verShape[0]*verShape[1]*verShape[2]*verShape[3], sizeof(float), verOutputFile);
    fclose(verOutputFile);
    verOutputFile = nullptr;
    return APP_ERR_OK;
}

APP_ERROR PVNet::Process(const std::string &imgPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }


    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);

    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    ret = SaveResult(imgPath, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

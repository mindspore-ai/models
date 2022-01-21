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
 * ============================================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "DCGAN.h"

InitParam initParam_;

APP_ERROR DCGAN::Init(const InitParam &initParam) {
    // Param init
    initParam_ = initParam;

    // Device init
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }

    // Context init
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }

    // Model init
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    // create random number from normal distribution (mean=0.0, std=1.0)
    std::mt19937 gen_{1213};
    std::normal_distribution<float> dis_{0.0, 1.0};

    return ret;
}

APP_ERROR DCGAN::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();

    return APP_ERR_OK;
}

APP_ERROR DCGAN::CreateRandomTensorBase(std::vector<MxBase::TensorBase> &inputs) {
    MxBase::TensorBase tensorBase;
    size_t D0 = initParam_.batchSize, D1 = 100, D2 = 1, D3 = 1;  // D0:batchsize
    const uint32_t dataSize = D0 * D1 * D2 * D3 * FLOAT32_TYPE_BYTE_NUM;

    float *mat_data = new float[dataSize / FLOAT32_TYPE_BYTE_NUM];
    for (size_t d0 = 0; d0 < D0; d0++) {
        for (size_t d1 = 0; d1 < D1; d1++) {
            for (size_t d2 = 0; d2 < D2; d2++) {
                for (size_t d3 = 0; d3 < D3; d3++) {
                    int i = d0 * D1 * D2 * D3 + d1 * D2 * D3 + d2 * D3 + d3;
                    mat_data[i] = dis_(gen_);
                }
            }
        }
    }

    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, initParam_.deviceId);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(mat_data),
                                     dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {static_cast<uint32_t>(D0), static_cast<uint32_t>(D1),
                                   static_cast<uint32_t>(D2), static_cast<uint32_t>(D3)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);

    inputs.push_back(tensorBase);

    return APP_ERR_OK;
}

APP_ERROR DCGAN::Inference(const std::vector<MxBase::TensorBase> &inputs,
                           std::vector<MxBase::TensorBase> &outputs) {
    // apply for output Tensor buffer
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        // shape
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        // define tensor
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, initParam_.deviceId);
        // request memory
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }

    // dynamic information
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;

    // do inferrnce
    auto startTime = std::chrono::high_resolution_clock::now();
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

APP_ERROR DCGAN::PostProcess(std::vector<MxBase::TensorBase> outputs, std::vector<cv::Mat> &resultMats) {
    APP_ERROR ret;
    ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
    }

    float *outputPtr = reinterpret_cast<float *>(outputs[0].GetBuffer());

    size_t H = initParam_.imageHeight, W = initParam_.imageWidth, C = CHANNEL;

    for (uint32_t b = 0; b < initParam_.batchSize; b++) {
        cv::Mat resultMat(initParam_.imageHeight, initParam_.imageWidth, CV_8UC3);
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    float *tmpLoc = outputPtr + b * C * H * W + (C - c - 1) * H * W + h * W + w;
                    // denormalize
                    float tmpNum = (*tmpLoc) * NORMALIZE_STD + NORMALIZE_MEAN;
                    // NCHW to NHWC
                    resultMat.at<cv::Vec3b>(h, w)[c] = static_cast<int>(tmpNum);
                }
            }
        }
        resultMats.push_back(resultMat);
    }

    return ret;
}

APP_ERROR DCGAN::SaveResult(std::vector<cv::Mat> &resultMats, const std::string &imgName) {
    DIR *dirPtr = opendir(initParam_.savePath.c_str());
    if (dirPtr == nullptr) {
        std::string path = "mkdir -p " + initParam_.savePath;
        system(path.c_str());
    }
    for (uint32_t b = 0; b < initParam_.batchSize; b++) {
        std::string file_path = initParam_.savePath + "/" + imgName + "-" + std::to_string(b) + ".jpg";
        cv::imwrite(file_path, resultMats[b]);
        std::cout << "[INFO] image saved path: " << file_path << std::endl;
    }

    return APP_ERR_OK;
}

APP_ERROR DCGAN::Process(uint32_t gen_id) {
    APP_ERROR ret;

    // create random tensor
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    ret = CreateRandomTensorBase(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    // do inference
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::cout << "[INFO] Inference finished!" << std::endl;

    // do postprocess
    std::vector<cv::Mat> resultMats = {};
    ret = PostProcess(outputs, resultMats);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    std::cout << "[INFO] Postprocess finished!" << std::endl;

    // save results
    std::string imgName = std::to_string(gen_id);
    ret = SaveResult(resultMats, imgName);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    std::cout << "[INFO] Result saved successfully!" << std::endl;

    return APP_ERR_OK;
}

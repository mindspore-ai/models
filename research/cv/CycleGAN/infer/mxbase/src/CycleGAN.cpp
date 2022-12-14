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

#include "CycleGAN.h"

CycleGAN::CycleGAN() {
    originalWidth_ = 256;
    originalHeight_ = 256;
}

APP_ERROR CycleGAN::Init(const InitParam &initParam) {
    // Param init
    initParam_ = initParam;

    // Device init
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }

    // Context init
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam_.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }

    // Model init
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam_.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return ret;
}

APP_ERROR CycleGAN::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();

    return APP_ERR_OK;
}

APP_ERROR CycleGAN::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    originalWidth_ = imageMat.cols;
    originalHeight_ = imageMat.rows;
    return APP_ERR_OK;
}

APP_ERROR CycleGAN::Resize(cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    cv::resize(srcImageMat, dstImageMat, cv::Size(initParam_.imageWidth, initParam_.imageHeight));
    return APP_ERR_OK;
}

APP_ERROR CycleGAN::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    size_t H = initParam_.imageHeight, W = initParam_.imageWidth, C = CHANNEL;
    const uint32_t dataSize = H * W * C;

    // mat NHWC to NCHW, BGR to RGB, and Normalize
    float *mat_data = new float[dataSize];
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                int i = (C - c - 1) * (H * W) + h * W + w;
                mat_data[i] = (imageMat.at<cv::Vec3b>(h, w)[c] - NORMALIZE_MEAN) / NORMALIZE_STD;
            }
        }
    }

    MxBase::MemoryData memoryDataDst(dataSize * 4, MxBase::MemoryData::MEMORY_DEVICE, initParam_.deviceId);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(&mat_data[0]),
                                     dataSize * 4, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.rows),
                                   static_cast<uint32_t>(imageMat.cols)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);

    return APP_ERR_OK;
}

APP_ERROR CycleGAN::Inference(const std::vector<MxBase::TensorBase> &inputs,
                              std::vector<MxBase::TensorBase> &outputs) {
    // apply for output Tensor buffer
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, initParam_.deviceId);
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

APP_ERROR CycleGAN::PostProcess(std::vector<MxBase::TensorBase> outputs, cv::Mat *resultMat) {
    APP_ERROR ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
    }
    float *outputPtr = reinterpret_cast<float *>(outputs[0].GetBuffer());

    size_t H = initParam_.imageHeight, W = initParam_.imageWidth, C = CHANNEL;

    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                float tmpNum = *(outputPtr + (C - c - 1) * (H * W) + h * W + w) * NORMALIZE_STD + NORMALIZE_MEAN;
                resultMat->at<cv::Vec3b>(h, w)[c] = static_cast<int>(tmpNum);
            }
        }
    }

    return APP_ERR_OK;
}

APP_ERROR CycleGAN::SaveResult(const cv::Mat &resultMat, const std::string &imgName) {
    DIR *dirPtr = opendir(initParam_.savePath.c_str());
    if (dirPtr == nullptr) {
        std::string path1 = "mkdir -p " + initParam_.savePath;
        system(path1.c_str());
    }

    std::string file_path = initParam_.savePath + "/" + imgName;
    cv::imwrite(file_path, resultMat);
    std::cout << "[INFO] image saved to: " << file_path << std::endl;
    return APP_ERR_OK;
}

APP_ERROR CycleGAN::Process(const std::string &imgPath, const std::string &imgName) {
    APP_ERROR ret;

    // read image as CV Mat
    cv::Mat imageMat;
    ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    // resize image
    ret = Resize(imageMat, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }

    // transform CVMat image to tensor
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    // do inference
    inputs.push_back(tensorBase);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::cout << "[INFO] Inference finished!" << std::endl;

    // do postprocess
    cv::Mat resultMat(initParam_.imageHeight, initParam_.imageWidth, CV_8UC3);
    ret = PostProcess(outputs, &resultMat);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    std::cout << "[INFO] Postprocess finished!" << std::endl;

    // save results
    cv::resize(resultMat, resultMat, cv::Size(originalWidth_, originalHeight_));
    ret = SaveResult(resultMat, imgName);
    if (ret != APP_ERR_OK) {
        LogError << "Save result failed, ret=" << ret << ".";
        return ret;
    }
    std::cout << "[INFO] Result saved successfully!" << std::endl;

    return APP_ERR_OK;
}

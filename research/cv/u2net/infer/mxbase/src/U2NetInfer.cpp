/*
 * Copyright 2022. Huawei Technologies Co., Ltd.
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
#include "U2NetInfer.h"
#include <sys/stat.h>
#include <unistd.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR U2NetInfer::Init(const InitParam &initParam) {
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

APP_ERROR U2NetInfer::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR U2NetInfer::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR U2NetInfer::Resize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat,
                             uint32_t target_h, uint32_t target_w) {
    cv::resize(srcImageMat, dstImageMat, cv::Size(target_h, target_w));
    return APP_ERR_OK;
}

APP_ERROR U2NetInfer::CVMatToTensorBase(const cv::Mat &imageMat,
                                        MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size();
             ++j) {
            shape.push_back((uint32_t)modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }

    // mat NHWC to NCHW, BGR to RGB
    size_t H = 320, W = 320, C = 3;

    float *mat_data = new float[dataSize];
    dataSize = dataSize * 4;
    int id;
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                id = (C - c - 1) * (H * W) + h * W + w;
                float tmp = imageMat.at<cv::Vec3b>(h, w)[c] / 255.0;
                if (c == 2) {
                    tmp = (tmp - 0.485) / 0.229;
                } else if (c == 1) {
                    tmp = (tmp - 0.456) / 0.224;
                } else {
                    tmp = (tmp - 0.406) / 0.225;
                }
                mat_data[id] = tmp;
            }
        }
    }

    MxBase::MemoryData memoryDataDst(
        dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(&mat_data[0]),
                                     dataSize,
                                     MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret =
        MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, 3, 320, 320};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape,
                                    MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR U2NetInfer::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size();
             ++j) {
            shape.push_back(
                (uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i],
                                  MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                                  deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    dynamicInfo.batchSize = 1;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs =
        std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR U2NetInfer::PostProcess(std::vector<MxBase::TensorBase> &outputs,
                                  cv::Mat &resultImg, uint32_t original_h,
                                  uint32_t original_w) {
    APP_ERROR ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
    }
    float *outputPtr = reinterpret_cast<float *>(outputs[0].GetBuffer());

    size_t H = 320, W = 320, org_H = resultImg.rows, org_W = resultImg.cols;

    cv::Mat outputImg(H, W, CV_8UC3);

    float tmpNum;
    for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
            tmpNum = *(outputPtr + h * W + w) * 255;
            outputImg.at<cv::Vec3b>(h, w)[0] = static_cast<int>(tmpNum);
        }
    }
    for (size_t h = 0; h < org_H; h++) {
        for (size_t w = 0; w < org_W; w++) {
            resultImg.at<cv::Vec3b>(h, w)[0] = outputImg.at<cv::Vec3b>(h, w)[0];
        }
    }
    Resize(resultImg, resultImg, original_h, original_w);
    return APP_ERR_OK;
}

APP_ERROR U2NetInfer::Process(const std::string &imgPath,
                              const std::string &outputPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    uint32_t original_h = static_cast<uint32_t>(imageMat.cols);
    uint32_t original_w = static_cast<uint32_t>(imageMat.rows);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    ret = Resize(imageMat, imageMat, 320, 320);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }

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
    cv::Mat resultImg(320, 320, CV_8UC3);
    ret = PostProcess(outputs, resultImg, original_h, original_w);

    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    ret = SaveResult(resultImg, imgPath, outputPath);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR U2NetInfer::SaveResult(cv::Mat resultImg, const std::string &imgPath,
                                 const std::string &outputPath) {
    std::string fileName = imgPath.substr(imgPath.find_last_of("/") + 1);
    LogInfo << "Finished inferencing " << fileName << ".";
    size_t dot = fileName.find_last_of(".");
    fileName = fileName.substr(0, dot);
    std::string output_path = outputPath + "/" + fileName + ".png";

    DIR *dirPtr = opendir(outputPath.c_str());
    if (dirPtr == nullptr) {
        std::string path1 = "mkdir -p " + outputPath;
        system(path1.c_str());
    }
    std::vector<cv::Mat> SrcMatpart(resultImg.channels());
    cv::split(resultImg, SrcMatpart);
    cv::imwrite(output_path, SrcMatpart[0]);
    return APP_ERR_OK;
}

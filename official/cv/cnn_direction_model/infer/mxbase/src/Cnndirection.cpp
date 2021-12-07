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

#include "Cnndirection.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <queue>
#include <utility>
#include <fstream>
#include <map>
#include <iostream>
#include "acl/acl.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

namespace {
    uint16_t READ_IMAGE_HEIGHT = 64;
    uint16_t READ_IMAGE_WIDTH = 64;
    uint16_t RESIZE_IMAGE_HEIGHT = 64;
    uint16_t RESIZE_IMAGE_WIDTH = 512;
    uint16_t FLOAT_SIZE = 4;
}  // namespace

APP_ERROR cnndirection::Init(const InitParam &initParam) {
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
    model_cnndirection = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_cnndirection->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR cnndirection::DeInit() {
    model_cnndirection->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR cnndirection::ReadImage(const std::string &imgPath, std::vector<std::vector<std::vector<float>>> &im) {
    cv::Mat imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::resize(imageMat, imageMat, cv::Size(READ_IMAGE_HEIGHT, READ_IMAGE_WIDTH));


    int w = imageMat.cols;
    int h = imageMat.rows;

    std::vector<float> rgb;
    for (int i = 0; i < h; i++) {
        std::vector<std::vector<float>> t;
        im.push_back(t);
        for (int j = 0; j < w; j++) {
            float r = imageMat.at<cv::Vec3b>(i, j)[2];
            float g =  imageMat.at<cv::Vec3b>(i, j)[1];
            float b =  imageMat.at<cv::Vec3b>(i, j)[0];
            rgb = {r, g, b};

            im[i].push_back(rgb);
        }
    }

    return APP_ERR_OK;
}

APP_ERROR cnndirection::ReSize(const std::vector<std::vector<std::vector<float>>> &image,
                               std::vector<std::vector<std::vector<float>>> &image_after) {
    std::vector<std::vector<std::vector<float>>> image_resize;
    static constexpr uint32_t Height = 64;
    static constexpr uint32_t Width = 64;

    for (int x = 0; x < RESIZE_IMAGE_HEIGHT; x++) {
            std::vector<std::vector<float>> tt;
            image_resize.push_back(tt);
            for (int y = 0; y < RESIZE_IMAGE_WIDTH; y++) {
                std::vector<float> tmp;
                if (x < READ_IMAGE_HEIGHT && y < READ_IMAGE_WIDTH) {
                    std::vector<float> t = {image[x][y][0]/127.5-1.0,
                                            image[x][y][1]/127.5-1.0, image[x][y][2] / 127.5-1.0};
                    tmp = t;
                } else {
                    std::vector<float> t = {1, 1, 1};
                    tmp = t;
                }
                image_resize[x].push_back(tmp);
            }
    }
    image_after = image_resize;
    return APP_ERR_OK;
}

APP_ERROR cnndirection::VectorToTensorBase_float(const std::vector<std::vector<std::vector<float>>> &batchFeatureVector,
                                    MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 1;
    std::vector<uint32_t> shape = {};
    shape.push_back(1);
    shape.push_back(batchFeatureVector[0][0].size());
    shape.push_back(batchFeatureVector.size());
    shape.push_back(batchFeatureVector[0].size());

    for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    float *metaFeatureData = new float[dataSize];
    uint32_t idx = 0;
    for (size_t ch = 0; ch < batchFeatureVector[0][0].size(); ch++) {
        for (size_t h = 0; h < batchFeatureVector.size(); h++) {
              for (size_t w = 0; w < batchFeatureVector[0].size(); w++) {
                metaFeatureData[idx++] = batchFeatureVector[h][w][ch];
                }
        }
    }
    MxBase::MemoryData memoryDataDst(dataSize * FLOAT_SIZE, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(metaFeatureData, dataSize * FLOAT_SIZE, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}


APP_ERROR cnndirection::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_cnndirection->GetOutputDataType();
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
    APP_ERROR ret = model_cnndirection->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference cnndirection failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR cnndirection::Process(const std::string &image_path, const InitParam &initParam, std::vector<int> &outputs) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs_tb = {};
    std::string infer_result_path = "./infer_results.txt";

    std::vector<std::vector<std::vector<float>>> image;
    ReadImage(image_path, image);
    ReSize(image, image);
    MxBase::TensorBase tensorBase;
    APP_ERROR ret = VectorToTensorBase_float(image, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "ToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret1 = Inference(inputs, outputs_tb);

    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret1 != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret1 << ".";
        return ret1;
    }

    if (!outputs_tb[0].IsHost()) {
            outputs_tb[0].ToHost();
        }
    float *value = reinterpret_cast<float *>(outputs_tb[0].GetBuffer());

    float res0 = value[0];
    float res1 = value[1];

    std::ofstream outfile(infer_result_path, std::ios::app);

    if (outfile.fail()) {
    LogError << "Failed to open result file: ";
    return APP_ERR_COMM_FAILURE;
    }
    outfile << res0<< "\t"<< res1<< "\n";
    outfile.close();

    if (res0 > res1) {
        outputs.push_back(0);
    } else {
        outputs.push_back(1);
    }
}

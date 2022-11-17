/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "SphereFaceOpencv.h"

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

void SphereFaceOpencv::getfilename(std::string *filename, const std::string &imgpath) {
    int i;
    for (i = imgpath.length() - 1; i >= 0; i--) {
    // '/' is the delimiter between the file naem and the parent directory in imgpath
        if (imgpath[i] == '/') {
            break;
        }
    }
    // '.' is the delimiter between the file name and the file suffix
    while (imgpath[++i] != '.') {
        *filename += imgpath[i];
    }
}

APP_ERROR SphereFaceOpencv::Init(const InitParam &initParam) {
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

APP_ERROR SphereFaceOpencv::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR SphereFaceOpencv::ReadImage(const std::string &imgPath,
                                   cv::Mat *imageMat) {
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR SphereFaceOpencv::ResizeImage(const cv::Mat &srcImageMat,
                                     uint8_t (&imageArray)[IMG_H][IMG_W][IMG_C]) {
    // HWC->CHW
    uint8_t image_chw[IMG_C][IMG_H][IMG_W];
    for (int i = 0; i < IMG_H; i++) {
        for (int j = 0; j < IMG_W; j++) {
            cv::Vec3b nums = srcImageMat.at<cv::Vec3b>(i, j);
            image_chw[0][i][j] = nums[0];
            image_chw[1][i][j] = nums[1];
            image_chw[2][i][j] = nums[2];
        }
    }
    // reshape
    uint32_t cnt = 0;
    for (int i = 0; i < IMG_C; i++) {
        for (int j = 0; j < IMG_H; j++) {
            for (int k = 0; k < IMG_W; k++) {
                imageArray[cnt / (IMG_W * 3)][(cnt / 3) % IMG_W][cnt % 3] =
                image_chw[i][j][k];
                cnt++;
            }
        }
    }
    // array to mat
    cv::Mat imgMat(IMG_H, IMG_W, CV_8UC3, imageArray);
    // encode
    std::vector<unsigned char> inImage;
    cv::imencode(".jpg", imgMat, inImage);

    cv::Mat img_decode = cv::imdecode(inImage, cv::IMREAD_COLOR);
    for (int i = 0; i < IMG_H; i++) {
        for (int j = 0; j < IMG_W; j++) {
        cv::Vec3b nums = img_decode.at<cv::Vec3b>(i, j);
        imageArray[i][j][0] = nums[0];
        imageArray[i][j][1] = nums[1];
        imageArray[i][j][2] = nums[2];
        }
    }

    return APP_ERR_OK;
}

APP_ERROR SphereFaceOpencv::NormalizeImage(
    uint8_t (&imageArray)[IMG_H][IMG_W][IMG_C],
    float (&imageArrayNormal)[IMG_H][IMG_W][IMG_C]) {
        for (int i = 0; i < IMG_H; i++) {
            for (int j = 0; j < IMG_W; j++) {
                for (int k = 0; k < IMG_C; k++) {
                    imageArrayNormal[i][j][k] = (imageArray[i][j][k] - 127.5) / 127.5;
                }
            }
        }
    return APP_ERR_OK;
}

APP_ERROR SphereFaceOpencv::ArrayToTensorBase(float (&imageArray)[IMG_H][IMG_W][IMG_C],
                                           MxBase::TensorBase *tensorBase) {
    const uint32_t dataSize = IMG_H * IMG_W * IMG_C * sizeof(float);
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE,
                                   deviceId_);
    MxBase::MemoryData memoryDataSrc(imageArray, dataSize,
                                   MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret =
        MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {IMG_C, IMG_H, IMG_W};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape,
                                   MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR SphereFaceOpencv::Inference(
    const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i],
                                MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                                deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        (*outputs).push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime)
                        .count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR SphereFaceOpencv::SaveResult(MxBase::TensorBase *tensor,
                                    const std::string &resultpath) {
    std::ofstream outfile(resultpath);
    APP_ERROR ret = (*tensor).ToHost();
    if (ret != APP_ERR_OK) {
        LogError << "ToHost failed";
        return ret;
    }

    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }

    float *result = reinterpret_cast<float *>((*tensor).GetBuffer());
    for (int i = 0; i < FEATURE_NUM; i++) {
        outfile << float(result[i]) << " ";
    }
    outfile << std::endl;
    outfile.close();
    return APP_ERR_OK;
}

APP_ERROR SphereFaceOpencv::Process(const std::string &imgPath,
                                 const std::string &resultPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, &imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    uint8_t img_array[IMG_H][IMG_W][IMG_C];
    ResizeImage(imageMat, img_array);
    float img_normalize[IMG_H][IMG_W][IMG_C];
    NormalizeImage(img_array, img_normalize);

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    MxBase::TensorBase tensorBase;
    ret = ArrayToTensorBase(img_normalize, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "ArrayToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);

    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime)
                      .count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::string filename = "";
    getfilename(&filename, imgPath);
    std::string resultpath = resultPath + "/" + filename + ".txt";
    ret = SaveResult(&outputs[0], resultpath);
    if (ret != APP_ERR_OK) {
        LogError << "SaveResult failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

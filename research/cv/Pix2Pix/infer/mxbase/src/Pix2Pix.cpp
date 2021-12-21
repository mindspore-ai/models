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
#include<algorithm>
#include "Pix2Pix.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
using MxBase::TensorBase;
using MxBase::MemoryData;
using MxBase::MemoryHelper;
using MxBase::TENSOR_DTYPE_FLOAT32;
using MxBase::DynamicInfo;
using MxBase::DynamicType;

const float NORMALIZE_MEAN = 255 / 2;
const float NORMALIZE_STD = 255 / 2;
const int OUTPUT_HEIGHT = 256;
const int OUTPUT_WIDTH = 256;
const int OUTPUT_CHANNEL = 3;
const double MAX_PX_VALUE = 255.0;
const double MIN_PX_VALUE = 0.0;


void PrintTensorShape(const std::vector<MxBase::TensorDesc> &tensorDescVec, const std::string &tensorName) {
    LogInfo << "The shape of " << tensorName << " is as follows:";
    for (size_t i = 0; i < tensorDescVec.size(); ++i) {
        LogInfo << "  Tensor " << i << ":";

        for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
            LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
        }
    }
}

void PrintInputShape(const std::vector<MxBase::TensorBase> &input) {
    MxBase::TensorBase img = input[0];
    LogInfo << "  -------------------------input0 ";
    LogInfo << img.GetDataType();
    LogInfo << img.GetShape()[0] << ", " << img.GetShape()[1]
    << ", "  << img.GetShape()[2] << ", " << img.GetShape()[3];
    LogInfo << img.GetSize();
}


APP_ERROR Pix2Pix::Init(const InitParam &initParam) {
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
    savePath_ = initParam.savePath;
    PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
    PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");

    return APP_ERR_OK;
}

APP_ERROR Pix2Pix::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Pix2Pix::ReadImage(const std::string &imgPath, cv::Mat *imageMat) {
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    LogInfo << "**********************************";
    LogInfo << "ReadImage() PATH = " << imgPath;
    LogInfo << "**********************************";
    LogInfo << "data" << imageMat->size();
    LogInfo << "******************************************************";
    LogInfo << "ReadImage() Image:";
    LogInfo << "imageMat.channels=" << imageMat->channels();
    LogInfo << "imageMat.cols=" << imageMat->cols;
    LogInfo << "imageMat.rows=" << imageMat->rows;
    LogInfo << "******************************************************";
    return APP_ERR_OK;
}

APP_ERROR Pix2Pix::CropImage(const cv::Mat &srcImageMat, cv::Mat *dstImageMat) {
    static cv::Rect rectOfImg(256, 0, 256, 256);
    *dstImageMat = srcImageMat(rectOfImg).clone();
    return APP_ERR_OK;
}

APP_ERROR Pix2Pix::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
    uint32_t dataSize = 1;
    for (size_t i = 0; i < modelDesc_.inputTensors.size(); ++i) {
        std::vector <uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.inputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t) modelDesc_.inputTensors[i].tensorDims[j]);
        }
        for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    }

    // mat NHWC to NCHW, BGR to RGB
    size_t H = 256, W = 256, C = 3;

    float *mat_data = new float[dataSize];
    dataSize = dataSize * 4;
    int id;
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                id = (C - c - 1) * (H * W) + h * W + w;
                mat_data[id] = (imageMat.at<cv::Vec3b>(h, w)[c] -
                (MAX_PX_VALUE - MIN_PX_VALUE) / 2.0) / ((MAX_PX_VALUE - MIN_PX_VALUE) / 2.0);
            }
        }
    }

    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(reinterpret_cast<void *>(&mat_data[0]), dataSize, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR  ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector <uint32_t> shape = {1, 3, 256, 256};
    *tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}


APP_ERROR Pix2Pix::Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs) {
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

    // PrintInputShape(inputs);

    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogInfo << "ModelInference success, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR Pix2Pix::PostProcess(std::vector<MxBase::TensorBase> outputs, cv::Mat *resultImg) {
    LogInfo << "output_size:" << outputs.size();
    LogInfo <<  "output0_datatype:" << outputs[0].GetDataType();
    LogInfo << "output0_shape:" << outputs[0].GetShape()[0] << ", "
    << outputs[0].GetShape()[1] << ", "  << outputs[0].GetShape()[2] << ", "
    << outputs[0].GetShape()[3];
    LogInfo << "output0_bytesize:"  << outputs[0].GetByteSize();

    APP_ERROR ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
    }

    float *outputPtr = reinterpret_cast<float *>(outputs[0].GetBuffer());

    size_t  H = OUTPUT_HEIGHT , W = OUTPUT_HEIGHT, C = OUTPUT_CHANNEL;

    cv::Mat outputImg(H, W, CV_8UC3);

    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                float tmpNum = *(outputPtr + (C - c - 1) * (H * W) + h * W + w) * (MAX_PX_VALUE - MIN_PX_VALUE)/2.0
                + (MAX_PX_VALUE - MIN_PX_VALUE)/2.0;
                outputImg.at<cv::Vec3b>(h, w)[c] = static_cast<int>(tmpNum);
            }
        }
    }
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                resultImg->at<cv::Vec3b>(h, w)[c] = outputImg.at<cv::Vec3b>(h, w)[c];
            }
        }
    }

    return APP_ERR_OK;
}

APP_ERROR Pix2Pix::SaveResult(const cv::Mat &resultImg, const std::string &imgName) {
    DIR *dirPtr = opendir(savePath_.c_str());
    if (dirPtr == nullptr) {
        std::string path1 = "mkdir -p " + savePath_;
        system(path1.c_str());
    }

    cv::imwrite(savePath_ + '/' + imgName, resultImg);
    return APP_ERR_OK;
}

APP_ERROR Pix2Pix::Process(const std::string &imgPath, const std::string &imgName) {
    cv::Mat imageMat;

    APP_ERROR ret = ReadImage(imgPath, &imageMat);

    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    cv::Mat cropImage;

    ret = CropImage(imageMat, &cropImage);
    if (ret != APP_ERR_OK) {
        LogError << "Crop failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "******************************************************";
    LogInfo << "After Crop Image:";
    LogInfo << "channels=" << cropImage.channels();
    LogInfo << "cols=" << cropImage.cols;
    LogInfo << "rows=" << cropImage.rows;
    LogInfo << "******************************************************";

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    MxBase::TensorBase tensorBase;
    APP_ERROR ret1 = CVMatToTensorBase(cropImage, &tensorBase);

    if (ret1 != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret1=" << ret1 << ".";
        return ret1;
    }

    LogInfo << "start push_back";
    inputs.push_back(tensorBase);
    LogInfo << inputs.size();
    LogInfo << "end push_back";

    auto startTime = std::chrono::high_resolution_clock::now();
    LogInfo << "start inference";
    ret = Inference(inputs, &outputs);
    LogInfo << "end inference";
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "-------------------------------------";
    LogInfo << outputs.size();
    LogInfo << "-------------------------------------";
    cv::Mat resultImg(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC3);
    ret = PostProcess(outputs, &resultImg);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    LogInfo << "------------------------------------------";
    LogInfo << "Postprocess success";
    LogInfo << resultImg.type();
    LogInfo << resultImg.channels();
    LogInfo << resultImg.cols;
    LogInfo << resultImg.rows;
    LogInfo << "------------------------------------------";

    ret = SaveResult(resultImg, imgName);
    LogInfo << "save mxbase image success";
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

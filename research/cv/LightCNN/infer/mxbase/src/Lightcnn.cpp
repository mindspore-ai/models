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
#include "Lightcnn.h"

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR Lightcnn::Init(const InitParam &initParam) {
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

APP_ERROR Lightcnn::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Lightcnn::ReadImage(const std::string &imgPath,
                                   cv::Mat *img) {
    *img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    return APP_ERR_OK;
}

APP_ERROR Lightcnn::ResizeImage(const cv::Mat &srcImageMat,
                                     float (&imageArray)[IMG_H][IMG_W]) {
    int a, b;
    int row = 128;
    int col = 128;
    for (a = 0; a < row; a++) {
         for (b = 0; b < col; b++) {
         imageArray[a][b] = srcImageMat.at<uchar>(a, b) / 255.0;
         }
    }
    return APP_ERR_OK;
}
APP_ERROR Lightcnn::ArrayToTensorBase(float (&imageArray)[IMG_H][IMG_W],
                                           MxBase::TensorBase *tensorBase) {
    const uint32_t dataSize = IMG_H * IMG_W * sizeof(float);
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageArray, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {IMG_H, IMG_W};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape,
                                   MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}
APP_ERROR Lightcnn::Inference(const std::vector<MxBase::TensorBase> &inputs,
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
APP_ERROR Lightcnn::SaveResult(MxBase::TensorBase *tensor,
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
    std::vector<uint32_t> out_shape = (*tensor).GetShape();
    LogError << "out_shape[1]: " << out_shape[1];
    for (int i = 0; i < out_shape[1]; i++) {
        outfile << float(result[i]) << " ";
    }
    outfile << std::endl;
    outfile.close();
    return APP_ERR_OK;
}

APP_ERROR Lightcnn::Process(const std::string &imgPath,
                                 const std::string &resultPath) {
    cv::Mat img;
    APP_ERROR ret = ReadImage(imgPath, &img);
    if (ret != APP_ERR_OK) {
      LogError << "ReadImage failed, ret=" << ret << ".";
      return ret;
    }
    float img_array[IMG_H][IMG_W];
    ResizeImage(img, img_array);

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    MxBase::TensorBase tensorBase;
    ret = ArrayToTensorBase(img_array, &tensorBase);
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
    ret = SaveResult(&outputs[1], resultpath);
    if (ret != APP_ERR_OK) {
      LogError << "SaveResult failed, ret=" << ret << ".";
      return ret;
    }
    return APP_ERR_OK;
}

void Lightcnn::getfilename(std::string *filename, const std::string &imgpath) {
    int i;
    for (i = imgpath.length() - 1; i >= 0; i--) {
      if (imgpath[i] == '/') {
        break;
      }
    }
    while (imgpath[++i] != '.') {
       *filename += imgpath[i];
    }
}

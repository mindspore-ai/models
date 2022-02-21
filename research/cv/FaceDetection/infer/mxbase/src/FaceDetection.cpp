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
#include "FaceDetection.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

// define the shape of tensor
uint32_t tensorDim0[] = {4, 84, 4};
uint32_t tensorDim1[] = {4, 84};
uint32_t tensorDim2[] = {4, 336, 4};
uint32_t tensorDim3[] = {4, 336};
uint32_t tensorDim4[] = {4, 1344, 4};
uint32_t tensorDim5[] = {4, 1344};

/**
 * initial dvppWrapper and model with parameters
 * @param initParam initialization parameters
 * @result status code of whether initialization is successful
 */
APP_ERROR FaceDetection::Init(const InitParam &initParam) {
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

/**
 * deinitialize model
 * @result status code of whether deinitialization is successful
 */
APP_ERROR FaceDetection::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

/**
 * get the image by name
 * @param imageName the name of the image to get
 * @param imageMat the Mat object that stores image
 * @param imgDr the path of imges
 * @result status code of whether read is successful
 */
APP_ERROR FaceDetection::ReadImage(const std::string &imageName, cv::Mat* imageMat, const std::string &imgDr) {
    std::string imgPath = imgDr + "/" + imageName + ".jpg";
    *imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(*imageMat, *imageMat, cv::COLOR_BGR2RGB);
    return APP_ERR_OK;
}

/**
 * get inferred results
 * @param inputs tensors that store information of image
 * @param outputs tensors that store inferred results
 * @result status code of whether inference is successful
 */
APP_ERROR FaceDetection::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                   std::vector<MxBase::TensorBase>* outputs) {
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
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/**
 * put the results of inference into bin file
 * @param outputs tensors that store inferred results
 * @param resultPathName the name of the file used to store the results
 * @result status code of whether writing is successful
 */
APP_ERROR FaceDetection::WriteResult2Bin(size_t index, MxBase::TensorBase output, std::string resultPathName) {
    FILE *fw = fopen(resultPathName.c_str(), "wb");
    if (fw == NULL) {
        LogError << "Failed to open result file: " << resultPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }

    // write inference result into file
    std::vector<std::vector<uint32_t>> dimList = {
        {tensorDim0[0], tensorDim0[1], tensorDim0[2]},
        {tensorDim1[0], tensorDim1[1]},
        {tensorDim2[0], tensorDim2[1], tensorDim2[2]},
        {tensorDim3[0], tensorDim3[1]},
        {tensorDim4[0], tensorDim4[1], tensorDim4[2]},
        {tensorDim5[0], tensorDim5[1]},
    };

    // check tensor is available
    std::vector<uint32_t> outputShape = output.GetShape();
    uint32_t len = outputShape.size();

    float *outputPtr = reinterpret_cast<float *>(output.GetBuffer());
    // three dimensional
    if (len == 4) {
        uint32_t H = dimList[index][1];  // H
        uint32_t W = dimList[index][2];  // w
        uint32_t C = dimList[index][0];  // C
        fwrite(outputPtr, sizeof(float), C * H * W, fw);
    } else {  // two dimensional
        uint32_t C = dimList[index][0];  // row
        uint32_t H = dimList[index][1];  // col
        fwrite(outputPtr, sizeof(float), C * H, fw);
    }
    fclose(fw);
    return APP_ERR_OK;
}

/**
 * put the results of inference into files
 * @param outputs tensors that store inferred results
 * @param imgName the image name used to name the folder
 * @param mxbaseResultPath the path where the results are stored
 * @result status code of whether writing is successful
 */
APP_ERROR FaceDetection::WriteResult(std::vector<MxBase::TensorBase> outputs, \
                                    const std::string &imgName, \
                                    const std::string &mxbaseResultPath) {
    std::string resultPathName = mxbaseResultPath;
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }
    resultPathName = "./mxbase_res/" + imgName;
    // create result directory when it does not exit
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create img output directory: " << resultPathName << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }
    // create result file under result directory
    resultPathName = resultPathName + "/output_";

    for (size_t  index = 0; index < outputs.size(); index++) {
        APP_ERROR ret = outputs[index].ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "tohost fail.";
            return ret;
        }
        WriteResult2Bin(index, outputs[index], resultPathName + std::to_string(index) + ".bin");
    }
    LogDebug << imgName + " done.";

    return APP_ERR_OK;
}

/**
 * resize and pad the image to the target size
 * @param imageMat images waiting to be resized and padded
 * @result status code of whether resize and padding is successful
 */
APP_ERROR FaceDetection::ResizeAndPadding(cv::Mat* imageMat) {
    int net_w = NET_WIDTH;
    int net_h = NET_HEIGHT;
    int im_w = imageMat->size().width;
    int im_h = imageMat->size().height;
    float scale;

    if (net_w == im_w && net_h == im_h) {
        return APP_ERR_OK;
    }

    // get scale
    if (static_cast<float>(im_w) / static_cast<float>(net_w) >= static_cast<float>(im_h) / static_cast<float>(net_h)) {
        scale = static_cast<float>(net_w) / static_cast<float>(im_w);
    } else {
        scale = static_cast<float>(net_h) / static_cast<float>(im_h);
    }
    if (scale != 1) {  // geometric scaling
        cv::resize(*imageMat, *imageMat, cv::Size(static_cast<int>(scale * im_w), \
                   static_cast<int>(scale * im_h)), 0, 0, cv::INTER_NEAREST);
        im_w = imageMat->size().width;
        im_h = imageMat->size().height;
    }

    // check whether padding is required
    if (im_w == net_w && im_h == net_h) {
        return APP_ERR_OK;
    }

    // padding to the target size
    float pad_w = fabs((net_w - im_w) / 2);
    float pad_h = fabs((net_h - im_h) / 2);

    int left = static_cast<int>(pad_w);
    int top = static_cast<int>(pad_h);
    int right = fabs((net_w - im_w)) - static_cast<int>(pad_w);
    int bottom = fabs((net_h - im_h)) - static_cast<int>(pad_h);

    copyMakeBorder(*imageMat, *imageMat, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));

    return APP_ERR_OK;
}

/**
 * convert HWC format to CHW format
 * @param imageMat images waiting for convert format
 * @param imageArray the array that stores image data in CHW format
 * @result status code of whether format conversion is successful
 */
APP_ERROR FaceDetection::hwc_to_chw(const cv::Mat &imageMat, float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH]) {
    // HWC->CHW
    for (int i = 0; i < NET_HEIGHT; i++) {
        for (int j = 0; j < NET_WIDTH; j++) {
            cv::Vec3b nums = imageMat.at<cv::Vec3b>(i, j);
            imageArray[0][i][j] = nums[0];
            imageArray[1][i][j] = nums[1];
            imageArray[2][i][j] = nums[2];
        }
    }
    return APP_ERR_OK;
}

/**
 * normalize the image data
 * @param imageArray the array that stores image data before normalize
 * @param imageArrayNormal the array that stores image data after normalize
 * @result status code of whether format normalization is successful
 */
APP_ERROR FaceDetection::NormalizeImage(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH], \
                                        float (&imageArrayNormal)[CHANNEL][NET_HEIGHT][NET_WIDTH]) {
    for (int i = 0; i < CHANNEL; i++) {
        for (int j = 0; j < NET_HEIGHT; j++) {
            for (int k = 0; k < NET_WIDTH; k++) {
                imageArrayNormal[i][j][k] = imageArray[i][j][k] / COLOR_RANGE;
            }
        }
    }
    return APP_ERR_OK;
}

/**
 * transfer data from array into the tensor
 * @param imageArray the array that stores image data
 * @param tensorBase the tensor that stores image data
 * @result status code of whether format normalization is successful
 */
APP_ERROR FaceDetection::ArrayToTensorBase(float (&imageArray)[CHANNEL][NET_HEIGHT][NET_WIDTH], \
                                           MxBase::TensorBase* tensorBase) {
    const uint32_t dataSize = CHANNEL * NET_HEIGHT * NET_WIDTH * sizeof(float);
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageArray, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {CHANNEL, NET_HEIGHT, NET_WIDTH};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR FaceDetection::Process(const std::string &imgName, \
                                 const std::string &imgDr, \
                                 const std::string &mxbaseResultPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgName, &imageMat, imgDr);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }

    // pre-process
    // step 1
    ResizeAndPadding(&imageMat);
    // step 2
    float img_array[CHANNEL][NET_HEIGHT][NET_WIDTH];
    hwc_to_chw(imageMat, img_array);
    // step 3
    float img_normalize[CHANNEL][NET_HEIGHT][NET_WIDTH];
    NormalizeImage(img_array, img_normalize);

    MxBase::TensorBase tensorBase;
    ret = ArrayToTensorBase(img_normalize, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "ArrayToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(tensorBase);
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    ret = WriteResult(outputs, imgName, mxbaseResultPath);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    imageMat.release();
    return APP_ERR_OK;
}

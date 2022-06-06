/*
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
 */
#include "RAS.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR RAS::Init(const InitParam &initParam) {
    this->deviceId_ = initParam.deviceId;
    this->outputDataPath_ = initParam.outputDataPath;
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

    this->model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = this->model_->Init(initParam.modelPath, this->modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    uint32_t input_data_size = 1;
    for (size_t j = 0; j < this->modelDesc_.inputTensors[0].tensorDims.size(); ++j) {
        this->inputDataShape_[j] = (uint32_t)this->modelDesc_.inputTensors[0].tensorDims[j];
        input_data_size *= this->inputDataShape_[j];
    }
    this->inputDataSize_ = input_data_size;

    return APP_ERR_OK;
}

APP_ERROR RAS::DeInit() {
    this->model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR RAS::ReadImage(const std::string &imgPath,
                                   cv::Mat &img) {
    img = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    return APP_ERR_OK;
}

APP_ERROR RAS::ImageLoader(const std::string &imgPath,
                                   cv::Mat &img) {
    img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    return APP_ERR_OK;
}

APP_ERROR RAS::PreProcess(const std::string &imgPath,
                          cv::Mat &img, float (&img_array)[IMG_C][IMG_H][IMG_W]) {
    cv::Size dsize = cv::Size(IMG_W, IMG_H);

    APP_ERROR ret = ReadImage(imgPath, img);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed.";
        return ret;
    }
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);
    cv::resize(img, img, dsize);
    float mean_[3] = {0.485, 0.456, 0.406};
    float std_[3] = {0.229, 0.224, 0.225};
    for (int i = 0; i < IMG_H; i++) {
        for (int j = 0; j < IMG_W; j++) {
            img.at<cv::Vec3f>(i, j)[0] = (img.at<cv::Vec3f>(i, j)[0] - mean_[0]) / std_[0];
            img.at<cv::Vec3f>(i, j)[0] = (img.at<cv::Vec3f>(i, j)[0] - mean_[0]) / std_[0];
            img.at<cv::Vec3f>(i, j)[1] = (img.at<cv::Vec3f>(i, j)[1] - mean_[1]) / std_[1];
            img.at<cv::Vec3f>(i, j)[1] = (img.at<cv::Vec3f>(i, j)[1] - mean_[1]) / std_[1];
            img.at<cv::Vec3f>(i, j)[2] = (img.at<cv::Vec3f>(i, j)[2] - mean_[2]) / std_[2];
            img.at<cv::Vec3f>(i, j)[2] = (img.at<cv::Vec3f>(i, j)[2] - mean_[2]) / std_[2];
            img_array[0][i][j] = img.at<cv::Vec3f>(i, j)[0];
            img_array[1][i][j] = img.at<cv::Vec3f>(i, j)[1];
            img_array[2][i][j] = img.at<cv::Vec3f>(i, j)[2];
        }
    }
    return APP_ERR_OK;
}

APP_ERROR RAS::ArrayToTensorBase(float (&imageArray)[IMG_C][IMG_H][IMG_W],
                                           MxBase::TensorBase *tensorBase) {
    const uint32_t dataSize = IMG_C * IMG_H * IMG_W * sizeof(float);
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageArray, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {IMG_C, IMG_H, IMG_W};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape,
                                   MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR RAS::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                 std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = this->model_->GetOutputDataType();
    for (size_t i = 0; i < this->modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)this->modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, this->deviceId_);
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
    APP_ERROR ret = this->model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);

    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR RAS::ReadResult(void) {
    std::ifstream inFile;
    std::string resultPath = this->outputDataPath_ + "results.txt";
    inFile.open(resultPath, std::ios_base::in);
    if (inFile.fail()) {
        LogError << "Failed to open annotation file: " << resultPath;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::uint32_t num = 0;
    double F_example = 0;
    double avg_F = 0;
    while (1) {
        inFile >> F_example;
        if (inFile.eof() != 0) {break;}
        avg_F += F_example;
        num++;
    }
    avg_F /= num;

    LogInfo << "================    Fmeasure Calculate    ===============";
    LogInfo << " | Average Fmeasure : " << avg_F << ".";
    LogInfo << "=========================================================";

    return APP_ERR_OK;
}

APP_ERROR RAS::PostProccess(std::vector<MxBase::TensorBase> &outputs,
                            const std::string &imgPath, const std::string &fileName) {
    cv::Mat img(IMG_W, IMG_H, CV_32FC1);
    cv::Mat img_gt = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    img_gt.convertTo(img_gt, CV_8UC1);
    if (!img_gt.data) {
        LogInfo << "Can not read image path : " << imgPath <<".";
        return -1;
    }
    APP_ERROR ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "to host fail.";
        return ret;
    }
    auto *netOutput = reinterpret_cast<float *>(outputs[0].GetBuffer());
    size_t data_size = outputs[0].GetSize();
    for (size_t i = 0; i < data_size; ++i) {
        img.at<float>(i) = static_cast<float>(*(reinterpret_cast<float *>(netOutput) + i));
    }

    cv::Size dsize = cv::Size(img_gt.cols, img_gt.rows);
    cv::resize(img, img, dsize);
    for (int i = 0; i < img_gt.rows * img_gt.cols; i++) {
        img.at<float>(i) = 1.0 / (1.0 + exp(-img.at<float>(i)));
    }
    cv::normalize(img, img, 0, 1.0, cv::NormTypes::NORM_MINMAX);
    cv::Mat save_img;
    cv::Mat predicts;
    img.convertTo(save_img, CV_8UC1, 1 * 255);
    cv::imwrite(this->outputDataPath_ + fileName, save_img);
    ImageLoader(this->outputDataPath_ + fileName, predicts);
    predicts.convertTo(predicts, CV_32FC1, 1.0 / 255);
    img_gt.convertTo(img_gt, CV_32FC1, 1.0 / 255);

    cv::Scalar img_mean;
    float sumLabel;
    img_mean = cv::mean(predicts);
    sumLabel = 2.0 * img_mean.val[0];
    if (sumLabel > 1) {
        sumLabel = 1.0;
    }
    float NumRec = 0;
    float NumAnd = 0;
    float num_obj = 0;
    float Precision = 0;
    float Recall = 0;
    float FmeasureF = 0;
    for (int i = 0; i < img_gt.rows * img_gt.cols; i++) {
        if (predicts.at<float>(i) >= sumLabel) {
            NumRec += 1.0;
            if (img_gt.at<float>(i) > 0.5) {
                NumAnd += 1.0;
            }
        }
        num_obj += img_gt.at<float>(i);
    }
    if (NumAnd != 0) {
        Precision = NumAnd / NumRec;
        Recall = NumAnd / num_obj;
        FmeasureF = (1.3 * Precision * Recall) / (0.3 * Precision + Recall);
    } else {
        Recall = 0;
        Precision = 0;
        FmeasureF = Precision + Recall;
    }

    std::ofstream fout;
    fout.open(this->outputDataPath_ + "results.txt", std::ios::app);
    fout << FmeasureF << std::endl;
    fout.close();

    LogInfo << "----------------------- Fmeasure ----------------------";
    LogInfo << "Fmeasure is : " << FmeasureF << ".";
    LogInfo << "-------------------------------------------------------";
    ReadResult();
    return APP_ERR_OK;
}


APP_ERROR RAS::Process(const std::string &inferPath, std::string &fileName) {
    cv::Mat img(IMG_W, IMG_H, CV_8UC3);
    float img_array[IMG_C][IMG_H][IMG_W];
    MxBase::TensorBase tensorBase;
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputIdsFile = inferPath + "images/" + fileName;
    int place = fileName.find(".jpg");
    if (place != -1) {
        fileName.replace(place, 4, ".png");
    }
    std::string gtInputFile = inferPath + "gts/" + fileName;

    APP_ERROR ret = PreProcess(inputIdsFile, img, img_array);
    if (ret != APP_ERR_OK) {
        LogError << "Preprocess failed, ret=" << ret << ".";
        return ret;
    }

    ret = ArrayToTensorBase(img_array, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "ArrayToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);
    std::vector<MxBase::TensorBase> outputs = {};
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    ret = PostProccess(outputs, gtInputFile, fileName);
    if (ret != APP_ERR_OK) {
        LogError << "PostProccess failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

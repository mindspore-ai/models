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
#include "FQA.h"
#include <math.h>
#include <dirent.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"

template<class Iter>
inline size_t argmax(Iter first, Iter last) {
    return std::distance(first, std::max_element(first, last));
}

APP_ERROR FQA::Init(const InitParam& initParam) {
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

APP_ERROR FQA::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR FQA::ReadImage(const std::string& imgPath, cv::Mat& imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);
    return APP_ERR_OK;
}

APP_ERROR FQA::ResizeImage(const cv::Mat& srcImageMat, float* transMat) {
    static constexpr uint32_t resizeHeight = 96;
    static constexpr uint32_t resizeWidth = 96;
    cv::Mat dstImageMat;
    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeHeight, resizeWidth));
    // convert NHWC to NCHW
    for (int i = 0; i < dstImageMat.rows; i++) {
        for (int j = 0; j < dstImageMat.cols; j++) {
            transMat[i*96+j] = static_cast<float>(dstImageMat.at<cv::Vec3b>(i, j)[0]) / 255;
            transMat[96*96+i*96+j] = static_cast<float>(dstImageMat.at<cv::Vec3b>(i, j)[1]) / 255;
            transMat[2*96*96+i*96+j] = static_cast<float>(dstImageMat.at<cv::Vec3b>(i, j)[2]) / 255;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR FQA::VectorToTensorBase(float* transMat, MxBase::TensorBase& tensorBase) {
    const uint32_t dataSize = 3*96*96*sizeof(float);
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(transMat, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {3, static_cast<uint32_t>(96*96)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR FQA::Inference(const std::vector<MxBase::TensorBase>& inputs, std::vector<MxBase::TensorBase>& outputs) {
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
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR FQA::PreProcess(
    const std::string& imgPath, std::vector<MxBase::TensorBase>& inputs, int& width, int& height) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    height = imageMat.cols;
    width = imageMat.rows;
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    float transMat[27648];
    ResizeImage(imageMat, transMat);
    MxBase::TensorBase tensorBase;
    ret = VectorToTensorBase(transMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "VectorToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);
    return APP_ERR_OK;
}

bool Readgt(const std::string& txtPath, float* eulgt, int (&kp_list)[5][2], int width, int height) {
    std::ifstream in(txtPath);
    if (!in) {
        return false;
    } else {
        for (int i = 0; i < 3; i++) {
            in >> eulgt[i];
        }
        for (int i = 0; i < 5; i++) {
            float x(0), y(0);
            in >> x >> y;
            if (x < 0 || y < 0) continue;
            kp_list[i][0] = static_cast<int>((x*96)/static_cast<float>(width));
            kp_list[i][1] = static_cast<int>((y*96)/static_cast<float>(height));
        }
    }
    return true;
}

int InferResult(float heatmap[5][48][48], int (&kp_coord_ori)[5][2]) {
    for (int i = 0; i < 5; i++) {
        std::vector<float> map_1(48*48, 0);
        float soft_sum(0);
        int o(0);
        for (int j = 0; j < 48; j++) {
            for (int k = 0; k < 48; k++) {
                map_1[o] = exp(heatmap[i][j][k]);
                o++;
                soft_sum += exp(heatmap[i][j][k]);
            }
        }
        for (int j = 0; j < 48*48; j++) {
            map_1[j] = map_1[j] / soft_sum;
        }
        int kp_coor = static_cast<int>(argmax(map_1.begin(), map_1.end()));
        kp_coord_ori[i][0] = (kp_coor % 48) * 2;
        kp_coord_ori[i][1] = (kp_coor / 48) * 2;
    }
    return -1;
}

int PrintResult(
    std::vector<std::vector<float>> kp_err, std::vector<std::vector<float>> eulers_err, std::vector<float> kp_ipn) {
    std::vector<float> kp_ave_error, euler_ave_error;
    for (uint32_t i = 0; i < kp_err.size(); i++) {
        float kp_error_all_sum(0);
        for (uint32_t j = 0; j < kp_err[i].size(); j++) {
            kp_error_all_sum += kp_err[i][j];
        }
        kp_ave_error.push_back(kp_error_all_sum/kp_err[i].size());
    }
    for (uint32_t i = 0; i < eulers_err.size(); i++) {
        float euler_ave_error_sum(0);
        for (uint32_t j = 0; j < eulers_err[i].size(); j++) {
            euler_ave_error_sum += eulers_err[i][j];
        }
        euler_ave_error.push_back(euler_ave_error_sum/eulers_err[i].size());
    }
    LogInfo << "================================== infer result ==================================\n";
    LogInfo << "5 keypoints average err: ["
            << kp_ave_error[0] << kp_ave_error[1] << kp_ave_error[2] << kp_ave_error[3] << kp_ave_error[4] << "]\n";
    LogInfo << "3 eulers average err: [" << euler_ave_error[0] << euler_ave_error[1] << euler_ave_error[2] << "]\n";
    float ipn(0), mae(0);
    for (uint32_t i = 0; i < kp_ipn.size(); i++) {
        ipn += kp_ipn[i];
    }
    LogInfo << "IPN of 5 keypoints: " << (ipn / kp_ipn.size()) * 100 << "\n";
    for (uint32_t i = 0; i < euler_ave_error.size(); i++) {
        mae += euler_ave_error[i];
    }
    LogInfo << "MAE of elur: " << mae / euler_ave_error.size() << "\n";
    LogInfo << "==================================================================================\n";
    return 0;
}

APP_ERROR FQA::Process(const std::string& testPath) {
    DIR* directory_pointer = NULL;
    struct dirent* entry;
    if ((directory_pointer = opendir(testPath.c_str())) == NULL) {
        printf("Error open\n");
        exit(0);
    }
    std::vector<std::vector<float>> kp_error_all(5), eulers_error_all(3);
    std::vector<float> kp_ipn;
    while ((entry = readdir(directory_pointer)) != NULL) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        std::string s = entry->d_name;
        if (s.substr(s.length()-3, 3) != "jpg") {
            continue;
        }
        std::string imgPath = testPath[testPath.length()-1] == '/' ? testPath+s : testPath+"/"+s;
        std::string txtPath = imgPath.substr(0, imgPath.length()-3) + "txt";
        std::vector<MxBase::TensorBase> inputs = {};
        std::vector<MxBase::TensorBase> outputs = {};
        int height(0), width(0);
        PreProcess(imgPath, inputs, width, height);
        APP_ERROR ret = Inference(inputs, outputs);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        MxBase::TensorBase& tensor0 = outputs[0];
        ret = tensor0.ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Tensor_0 deploy to host failed.";
            return ret;
        }
        auto eulers_ori = reinterpret_cast<float(*)>(tensor0.GetBuffer());
        eulers_ori[0] *= 90;
        eulers_ori[1] *= 90;
        eulers_ori[2] *= 90;
        MxBase::TensorBase& tensor1 = outputs[1];
        ret = tensor1.ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Tensor_1 deploy to host failed.";
            return ret;
        }
        auto heatmap = reinterpret_cast<float(*)[48][48]>(tensor1.GetBuffer());
        std::string savePath = "infer_result/" + s.substr(0, s.length()-3) + "txt";
        std::ofstream out(savePath);
        std::ifstream in(txtPath);
        float eulgt[3];
        int kp_list[5][2] = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}, kp_coord_ori[5][2];
        bool euler_kps_do = Readgt(txtPath, eulgt, kp_list, width, height);
        if (!euler_kps_do) {
            continue;
        }
        InferResult(heatmap, kp_coord_ori);
        out << "eulers_error:";
        for (int i = 0; i < 3; i++) {
            eulers_error_all[i].push_back(abs(eulers_ori[i] - eulgt[i]));
            out << " " << abs(eulers_ori[i] - eulgt[i]);
        }
        out << "\n";
        bool cur_flag = true;
        float eye_dis(1.0);
        if (kp_list[0][0] < 0 || kp_list[0][1] < 0 || kp_list[1][0] < 0 || kp_list[1][1] < 0) {
            cur_flag = false;
        } else {
            eye_dis = sqrt(
                pow(abs(kp_list[0][0] - kp_list[1][0]), 2) + pow(abs(kp_list[0][1] - kp_list[1][1]), 2));
        }
        float cur_error_sum(0), cnt(0);
        out << "keypoints_error:";
        for (int i = 0; i < 5; i++) {
            if (kp_list[i][0] != -1) {
                float dis = sqrt(
                    pow(kp_list[i][0] - kp_coord_ori[i][0], 2) +
                    pow(kp_list[i][1] - kp_coord_ori[i][1], 2));
                out << " " << dis;
                kp_error_all[i].push_back(dis);
                cur_error_sum += dis;
                cnt++;
            }
        }
        out << "\n";
        if (cur_flag) {
            kp_ipn.push_back(cur_error_sum / cnt / eye_dis);
        }
    }
    PrintResult(kp_error_all, eulers_error_all, kp_ipn);
    closedir(directory_pointer);
    return APP_ERR_OK;
}

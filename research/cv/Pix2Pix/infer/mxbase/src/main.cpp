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
#include <iostream>
#include <fstream>
#include <vector>
#include "Pix2Pix.h"
#include <opencv2/opencv.hpp>

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> *imgFiles) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << path << path.c_str();
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        imgFiles->emplace_back(fileName);
    }
    LogInfo << "opendir ok. dir:";
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as '../data/test_img'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../data/Pix2Pix_for_facades.om";
    initParam.savePath = "../data/mxbase_result";
    initParam.imgPath = "../data/test_img";

    auto pix2pix = std::make_shared<Pix2Pix>();
    APP_ERROR ret = pix2pix->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Pix2Pix init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    std::vector<std::string> imgFilePaths;
    ret = ScanImages(imgPath, &imgFilePaths);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    int imgNum = 1;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (auto &imgFile : imgFilePaths) {
        LogInfo << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        LogInfo << "imgFile = " << imgFile;
        LogInfo << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        ret = pix2pix->Process(imgPath + '/' + imgFile, imgFile);
        if (ret != APP_ERR_OK) {
            LogError << "Pix2Pix process failed, ret=" << ret << ".";
            pix2pix->DeInit();
            return ret;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    pix2pix->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgNum / pix2pix->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}

/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd
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
#include <dirent.h>
#include <fstream>
#include <string>
#include <iostream>
#include "PVNet.h"
#include "MxBase/Log/Log.h"


APP_ERROR ScanImages(const std::string &path, std::vector<std::string> &imgFiles) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        imgFiles.emplace_back(path + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

APP_ERROR ReadImgFiles(const std::string &txtPath, const std::string &datasetPath, std::vector<std::string> &imgFiles) {
    std::ifstream testTxt(txtPath.c_str());
    std::string fileName;
    if (testTxt) {
        while (getline(testTxt, fileName)) {
            imgFiles.emplace_back(datasetPath + "/" + fileName);
        }
    } else {
        LogError << "Open file failed. file:" << txtPath;
        return APP_ERR_INTERNAL_ERROR;
    }
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc <= 2) {
        LogWarn << "Please inputs test.txt and dataset path, such as '../data/cat/test.txt ../data/cat/images'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;

    initParam.modelPath = "./data/models/pvnet.om";
    auto pvnet = std::make_shared<PVNet>();
    APP_ERROR ret = pvnet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "PVNet init failed, ret=" << ret << ".";
        return ret;
    }

    std::string txtPath = argv[1];
    std::string datasetPath = argv[2];
    std::vector<std::string> imgFilePaths;
    ret = ReadImgFiles(txtPath, datasetPath, imgFilePaths);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    auto startTime = std::chrono::high_resolution_clock::now();
    for (auto &imgFile : imgFilePaths) {
        ret = pvnet->Process(imgFile);
        if (ret != APP_ERR_OK) {
            LogError << "PVNet process failed, ret=" << ret << ".";
            pvnet->DeInit();
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    pvnet->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgFilePaths.size() / pvnet->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}

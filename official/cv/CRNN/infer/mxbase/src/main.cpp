/**
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

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "CrnnRecognition.h"
#include "MxBase/Log/Log.h"

namespace CONST_CONFIG {
    const uint32_t CLASS_NUM = 37;
    const uint32_t OBJECT_NUM = 24;
    const uint32_t BLANK_INDEX = 36;
}

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> &imgFiles) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir: " << path;
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

void SetInitParam(InitParam &initParam) {
    initParam.deviceId = 0;
    initParam.classNum = CONST_CONFIG::CLASS_NUM;
    initParam.objectNum = CONST_CONFIG::OBJECT_NUM;
    initParam.blankIndex = CONST_CONFIG::BLANK_INDEX;
    initParam.labelPath = "../config/crnn.names";
    initParam.argmax = false;
    initParam.modelPath = "../model/crnn.om";
}

void ShowUsage() {
    std::cout << "Usage   : ./crnn <--image or --dir> [Option]" << std::endl;
    std::cout << "Options :" << std::endl;
    std::cout << " --image  the path of single infer image, such as ./crnn --image <path>/test.jpg." << std::endl;
    std::cout << " --dir   the dir of batch infer images, such as ./crnn --dir <path>/images." << std::endl;
    return;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Please use as follows." << std::endl;
        ShowUsage();
        return APP_ERR_OK;
    }

    std::string option = argv[1];
    std::string imgPath = argv[2];

    if (option != "--image" && option != "--dir") {
        std::cout << "Please use as follows." << std::endl;
        ShowUsage();
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    SetInitParam(initParam);

    auto crnn = std::make_shared<CrnnRecognition>();
    APP_ERROR ret = crnn->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "CrnnRecognition init failed, ret=" << ret << ".";
        return ret;
    }

    if (option == "--image") {
        std::ofstream infer_ret("infer_result_single.txt", std::ios::app);
        std::string result = "";
        ret = crnn->Process(imgPath, result);
        int file_pos = imgPath.find("//") + 1;
        infer_ret << imgPath.substr(file_pos, -1) << " " << result << std::endl;
        if (ret != APP_ERR_OK) {
            LogError << "CrnnRecognition process failed, ret=" << ret << ".";
            crnn->DeInit();
            return ret;
        }
        crnn->DeInit();
        infer_ret.close();
        return APP_ERR_OK;
    }

    std::ofstream infer_ret("infer_result_multi.txt");
    std::vector<std::string> imgFilePaths;
    ScanImages(imgPath, imgFilePaths);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (auto & imgFile : imgFilePaths) {
        std::string result = "";
        ret = crnn->Process(imgFile, result);
        int nPos = imgFile.find("//") + 2;
        std::string fileName = imgFile.substr(nPos, -1);
        infer_ret << fileName << " " << result << std::endl;
        if (ret != APP_ERR_OK) {
            LogError << "CrnnRecognition process failed, ret=" << ret << ".";
            crnn->DeInit();
            return ret;
        }
    }

    crnn->DeInit();
    auto endTime = std::chrono::high_resolution_clock::now();
    infer_ret.close();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgFilePaths.size() / crnn->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";

    std::ifstream infer_file("infer_result_multi.txt");
    int fCount = 0;
    float acc = 0;
    std::string s;
    while (getline(infer_file, s)) {
        int sPos = s.find(" ");
        std::string fileName = s.substr(0, sPos);
        int sPos1 = fileName.find_last_of("_");
        int sPos2 = fileName.find_last_of(".");
        std::string label = fileName.substr(sPos1+1, sPos2-sPos1-1);

        std::string inferRet = s.substr(sPos1+1, -1);
        transform(label.begin(), label.end(), label.begin(), ::tolower);

        std::string out;
        std::vector<std::string> str;

        std::istringstream divide(inferRet);
        while (divide >> out) {
            str.push_back(out);
        }

        if (label == str[1]) {
            acc++;
        }
        fCount++;
    }
    infer_file.close();
    std::cout << "hitted count is " << acc << ", label count is " << fCount << std::endl;
    std::cout << "infer accuracy is " << acc/fCount << std::endl;
    return APP_ERR_OK;
}

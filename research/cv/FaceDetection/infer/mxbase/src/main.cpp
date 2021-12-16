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

#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "FaceDetection.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

/**
 * define initialization parameters
 * @param initParam the pointer to the initialization parameter
 */
void InitFaceDetectionParam(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->modelPath = "../models/FaceDetection_mindspore.om";
    initParam->classNum = 1;
}

/**
 * get the names of all images under the path
 * @param path the path where the images are stored
 * @param imgNames vector that stores image names
 */
void getAllFilesName(const std::string &path, std::vector<std::string>* imgNames) {
    DIR *dir;
    struct dirent *ptr;
    if ((dir = opendir(path.c_str())) == NULL) {
        perror("Open dir error...");
        return;
    }

    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {    // current dir OR parent dir
            continue;
        } else if (ptr->d_type == 8) {  // file
            std::string strFileName;
            strFileName = ptr->d_name;
            imgNames->push_back(strFileName.substr(0, strFileName.size()-4));
        } else {
            continue;
        }
    }
    closedir(dir);
}

int main(int argc, char* argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path and output path, ";
        LogWarn << "such as './face_detection .././dataset_val/JPEGImages ./mxbase_res'.";
        return APP_ERR_OK;
    }
    std::string imgDir = argv[1];
    std::string mxbaseResultPath = argv[2];

    InitParam initParam;
    InitFaceDetectionParam(&initParam);
    auto faceDetection = std::make_shared<FaceDetection>();
    APP_ERROR ret = faceDetection->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FaceDetection init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::string> imgNames = {};
    getAllFilesName(imgDir, &imgNames);
    LogInfo << imgNames.size();
    for (std::string imgName : imgNames) {
        ret = faceDetection->Process(imgName, imgDir, mxbaseResultPath);
        if (ret != APP_ERR_OK) {
            LogError << "FaceDetection process failed, ret=" << ret << ".";
            faceDetection->DeInit();
            return ret;
        }
    }
    faceDetection->DeInit();

    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " img/sec.";

    return APP_ERR_OK;
}

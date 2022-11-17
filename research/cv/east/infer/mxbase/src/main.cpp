/*
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

#include <dirent.h>
#include <iostream>
#include <vector>
#include "EASTDetection.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void ShowUsage() {
    LogInfo << "Usage   : ./east <--image or --dir> [Option]" << std::endl;
    LogInfo << "Options :" << std::endl;
    LogInfo << " --image infer_image_path    the path of single infer image, such as "
               "./east --image /home/infer/images/test.jpg." << std::endl;
    LogInfo << " --dir infer_image_dir       the dir of batch infer images, such as "
               "./east --dir /home/infer/images." << std::endl;
    return;
}

APP_ERROR ReadImagesPath(const std::string& dir, std::vector<std::string> *imagesPath) {
    DIR *dirPtr = opendir(dir.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir: " << dir;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }
        (*imagesPath).emplace_back(dir + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LogInfo << "Please use as follows." << std::endl;
        ShowUsage();
        return APP_ERR_OK;
    }

    std::string option = argv[1];
    std::string imgPath = argv[2];

    if (option != "--image" && option != "--dir") {
        LogInfo << "Please use as follows." << std::endl;
        ShowUsage();
        return APP_ERR_OK;
    }

    InitParam initParam = {0, "0.2", "0.9", true, "../../data/models/east.om", 2};
    auto east = std::make_shared<EASTDetection>();
    APP_ERROR ret = east->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogInfo << "EASTDetectionOpencv init failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "End to Init east.";

    std::vector<std::string> imagesPath;
    if (option == "--image") {
        imagesPath.emplace_back(imgPath);
    } else {
        ret = ReadImagesPath(imgPath, &imagesPath);
    }

    if (ret != APP_ERR_OK) {
        LogInfo << "read file failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "read file success.";

    for (auto path : imagesPath) {
        LogInfo << "read image path " << path;
        east->Process(path);
    }

    east->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " images/sec.";
    return APP_ERR_OK;
}

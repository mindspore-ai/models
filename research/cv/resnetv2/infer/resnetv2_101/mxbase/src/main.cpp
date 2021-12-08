/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "Resnetv2.h"
#include "MxBase/Log/Log.h"


namespace {
    const uint32_t CLASS_NUM = 10;
}

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

        imgFiles->emplace_back(path + "/" + fileName);
    }
    LogInfo << "opendir ok. dir:";
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path and result path, such as './resnetv2 image_dir res_dir'";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../data/config/cifar10.names";
    initParam.topk = 5;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/resnetv2.om";
    auto resnetv2 = std::make_shared<Resnetv2>();
    APP_ERROR ret = resnetv2->Init(initParam);
    if (ret != APP_ERR_OK) {
        resnetv2->DeInit();
        LogError << "Resnetv2Classify init failed, ret = " << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    std::vector<std::string> imgFilePaths;
    ret = ScanImages(imgPath, &imgFilePaths);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    std::string resPath = argv[2];
    auto startTime = std::chrono::high_resolution_clock::now();
    for (auto &imgFile : imgFilePaths) {
        ret = resnetv2->Process(imgFile, resPath);
        if (ret != APP_ERR_OK) {
            LogError << "Resnetv2Classify process failed, ret = " << ret << ".";
            resnetv2->DeInit();
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    resnetv2->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgFilePaths.size() / resnetv2->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost:" << costMilliSecs << " ms\tfps: " << fps << "imgs/sec";
    return APP_ERR_OK;
}


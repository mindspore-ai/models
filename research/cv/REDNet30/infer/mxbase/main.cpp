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

#include "red30Superresolution.h"
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
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

// infer images
int main(int argc, char *argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path and result path, such as './red30 ./test.png ./result'";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../convert/red30.om";
    red30Superresolution red30SR;
    APP_ERROR ret = red30SR.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "red30Superresolution init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgPath = argv[1];
    std::string resultPath = argv[2];
    DIR *dirPtr = opendir(resultPath.c_str());

    // if it has no dir
    if (dirPtr == nullptr) {
        LogInfo << "opendir failed. so make dir" << resultPath.c_str();
        ret = mkdir(resultPath.c_str(), 0755);
        if (ret != 0) {
            return -1;
        }
    }
    std::vector<std::string> imgFilePaths;
    ret = ScanImages(imgPath, imgFilePaths);
    if (ret != APP_ERR_OK) {
        red30SR.DeInit();
        return ret;
    }
    for (auto &imgFile : imgFilePaths) {
        ret = red30SR.Process(imgFile, resultPath);
        if (ret != APP_ERR_OK) {
            LogError << "red30 process failed, ret=" << ret << ".";
            red30SR.DeInit();
            return ret;
        }
    }
    if (ret != APP_ERR_OK) {
        LogError << "red30Superresolution process failed, ret=" << ret << ".";
        red30SR.DeInit();
        return ret;
    }

    red30SR.DeInit();
    return APP_ERR_OK;
}

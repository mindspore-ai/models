// Mindx.cpp : Defines the entry point for the console application.
//

/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
#include "MxBase/Log/Log.h"
#include "LearningToSeeInTheDark.h"

namespace {
    const uint32_t BATCH_SIZE = 8;
    const uint32_t EACH_LABEL_LENGTH = 4;
    const uint32_t MAX_LENGTH = 12121088;
}  // namespace

APP_ERROR ReadFilesFromPath(const std::string &path, std::vector<std::string> *files) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "Opendir failed. dir:" << path << path.c_str();
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        files->emplace_back(path + "/" + fileName);
    }
    LogInfo << "Opendir ok. dir:";
    closedir(dirPtr);
    // sort ascending order
    std::sort(files->begin(), files->end());
    std::cout << "The size of files is " << files->size() << std::endl;
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as '../../data/images'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../../data/model/sid.om";
    auto learningtoseeinthedark = std::make_shared<LearningToSeeInTheDark>();
    APP_ERROR ret = learningtoseeinthedark->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "LearningToSeeInTheDark Classify init failed, ret=" << ret << ".";
        return ret;
    }

    std::string inferPath = argv[1];
    std::vector<std::string> files;
    ret = ReadFilesFromPath(inferPath, &files);
    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        return ret;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < files.size(); i++) {
        ret = learningtoseeinthedark->Process(files[i]);
        if (ret != APP_ERR_OK) {
            LogError << "LearningToSeeInTheDark Classify process failed, ret=" << ret << ".";
            learningtoseeinthedark->DeInit();
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    learningtoseeinthedark->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0*files.size() / learningtoseeinthedark->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost:" << costMilliSecs << " ms\tfps: " << fps << "imgs/sec";
    return APP_ERR_OK;
}

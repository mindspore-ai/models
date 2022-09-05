/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "DelfOpencv.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void InitShmParam(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->modelPath = "../out/delf.om";
}

APP_ERROR ReadFilesFromPath(const std::string &path, std::vector<std::string> *files) {
    DIR *dir = NULL;
    struct dirent *ptr = NULL;

    if ((dir=opendir(path.c_str())) == NULL) {
        LogError << "Open dir error: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }

    while ((ptr=readdir(dir)) != NULL) {
        if (ptr->d_type == 8) {
            files->push_back(ptr->d_name);
        }
    }
    closedir(dir);
    std::sort(files->begin(), files->end());

    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    // Init Param
    if (argc <= 1) {
        LogWarn << "Please input bin-file path, such as '../Preprocess_result/images_batch'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../out/delf.om";
    auto delf = std::make_shared<DelfOpencv>();
    APP_ERROR ret = delf->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Delf init failed, ret=" << ret << ".";
        return ret;
    }
    // read bin files
    std::string inferPath = argv[1];
    std::vector<std::string> files;
    ret = ReadFilesFromPath(inferPath, &files);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    // do infer
    LogInfo << "begin infer";
    for (uint32_t i = 0; i < files.size(); i++) {
        LogInfo << files[i];
        ret = delf->Process(inferPath, files[i]);
        if (ret != APP_ERR_OK) {
            LogError << "delf process failed, ret=" << ret << ".";
            delf->DeInit();
            return ret;
        }
    }
    LogInfo << "infer succeed and write the result data with binary file !";
    delf->DeInit();
//    auto startTime = std::chrono::high_resolution_clock::now();
//    int cnt = 0;
}

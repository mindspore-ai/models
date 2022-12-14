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
 * ============================================================================
 */

#include <sys/types.h>
#include <dirent.h>
#include <string.h>

#include <iostream>
#include <vector>

#include "CycleGAN.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void InitCycleGANParam(InitParam *initParam, char *argv[]) {
    initParam->deviceId = 0;
    initParam->checkTensor = true;
    initParam->modelPath = argv[1];
    initParam->dataPath = argv[2];
    initParam->savePath = argv[3];
    initParam->imageWidth = 256;
    initParam->imageHeight = 256;
}

void getFiles(std::string path, std::vector<std::string> &filenames) {
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str()))) {
        std::cout << "Folder doesn't Exist!" << std::endl;
        return;
    }
    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            filenames.push_back(ptr->d_name);
        }
    }
    closedir(pDir);
}

int main(int argc, char *argv[]) {
    InitParam initParam;
    InitCycleGANParam(&initParam, argv);

    std::vector<std::string> filenames;
    getFiles(initParam.dataPath, filenames);

    // Create cyclegan
    auto cyclegan = std::make_shared<CycleGAN>();

    // do init
    APP_ERROR ret = cyclegan->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "CycleGAN init failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "End to Init CycleGAN.";

    // get data
    for (int i = 0; i < filenames.size(); i++) {
        std::string filePath = initParam.dataPath + filenames[i];
        // do process
        LogInfo << "generate image " << i;
        ret = cyclegan->Process(filePath, filenames[i]);
        if (ret != APP_ERR_OK) {
            LogError << "CycleGAN process failed, ret=" << ret << ".";
            cyclegan->DeInit();
            return ret;
        }
    }

    // do deinit
    cyclegan->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " images/sec.";
    return APP_ERR_OK;
}

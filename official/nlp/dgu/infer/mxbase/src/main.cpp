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
#include "DGU.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;
uint32_t g_total = 0;
uint32_t g_total_acc = 0;

void InitDGUParam(InitParam* initParam, const std::string &taskName) {
    initParam->deviceId = 0;
    if (taskName == "atis_intent") {
        initParam->labelPath = "../data/config/map_tag_intent_id.txt";
        initParam->modelPath = "../data/model/atis_intent.om";
        initParam->classNum = 26;
    } else if (taskName == "mrda") {
        initParam->labelPath = "../data/config/map_tag_mrda_id.txt";
        initParam->modelPath = "../data/model/mrda.om";
        initParam->classNum = 5;
    } else if (taskName == "swda") {
        initParam->labelPath = "../data/config/map_tag_swda_id.txt";
        initParam->modelPath = "../data/model/swda.om";
        initParam->classNum = 42;
    }
}

APP_ERROR ReadFilesFromPath(const std::string &path, std::vector<std::string> *files) {
    DIR *dir = NULL;
    struct dirent *ptr = NULL;

    if ((dir=opendir(path.c_str())) == NULL) {
        LogError << "Open dir error: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }

    while ((ptr=readdir(dir)) != NULL) {
        // d_type == 8 is file
        int file_d_type = 8;
        if (ptr->d_type == file_d_type) {
            files->push_back(ptr->d_name);
        }
    }
    closedir(dir);
    // sort ascending order
    sort(files->begin(), files->end());
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './dgu /input/data/atis_intent 0 atis_intent'.";
        return APP_ERR_OK;
    }

    InitParam initParam;
    std::string taskName = argv[3];
    InitDGUParam(&initParam, taskName);
    auto dguBase = std::make_shared<DGUBase>();
    APP_ERROR ret = dguBase->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "dgu init failed, ret=" << ret << ".";
        return ret;
    }

    std::string inferPath = argv[1];
    std::vector<std::string> files;
    ret = ReadFilesFromPath(inferPath + "00_data", &files);
    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        return ret;
    }
    // do eval and calc the f1 score
    bool eval = atoi(argv[2]);
    for (uint32_t i = 0; i < files.size(); i++) {
        LogInfo << "read file name: " << files[i];
        ret = dguBase->Process(inferPath, files[i], eval);
        if (ret != APP_ERR_OK) {
            LogError << "dguBase process failed, ret=" << ret << ".";
            dguBase->DeInit();
            return ret;
        }
    }

    if (eval) {
        LogInfo << "==============================================================";
        if (g_total == 0) {
        LogInfo << "Infer total is 0.";
        } else {
            float acc = (g_total_acc * 1.0) / (g_total * 1.0);
            LogInfo << "Acc: " << acc;
        }
        LogInfo << "==============================================================";
    }
    dguBase->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    double scale = 1000.0;
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * scale / costSum << " bin/sec.";
    return APP_ERR_OK;
}

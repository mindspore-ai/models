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
#include "KTNetBase.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_infer_cost;

void InitKTNetParam(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->modelPath = "../data/model/KTNET_squad.om";
}

APP_ERROR ReadFilesFromPath(const std::string &path, std::vector<std::string> *files) {
    DIR *dir = NULL;
    struct dirent *ptr = NULL;
    int file_type = 8;
    if ((dir = opendir(path.c_str())) == NULL) {
        LogError << "Open dir error: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }

    while ((ptr = readdir(dir)) != NULL) {
        // d_type == 8 is file
        if (ptr->d_type == file_type) {
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
        LogWarn << "Please input image path, such as './ltney /input/data 0'.";
        return APP_ERR_OK;
    }

    InitParam initParam;
    InitKTNetParam(&initParam);
    auto ktnetBase = std::make_shared<KTNetBase>();
    APP_ERROR ret = ktnetBase->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "KTNetBase init failed, ret=" << ret << ".";
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
        ret = ktnetBase->Process(inferPath, files[i], eval);
        if (ret != APP_ERR_OK) {
            LogError << "KTNetBase process failed, ret=" << ret << ".";
            ktnetBase->DeInit();
            return ret;
        }
    }

    ktnetBase->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_infer_cost.size(); i++) {
        costSum += g_infer_cost[i];
    }
    double scale = 1000;
    LogInfo << "Infer items sum " << g_infer_cost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_infer_cost.size() * scale / costSum << " bin/sec.";
    return APP_ERR_OK;
}

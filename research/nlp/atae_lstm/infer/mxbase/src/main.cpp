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
#include "AtaeLstmBase.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void InitAtaeLstmParam(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->modelPath = "../data/model/atae_lstm.om";
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
        if (ptr->d_type == 8) {
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
        LogWarn << "Please input path.";
        return APP_ERR_OK;
    }

    // init model
    InitParam initParam;
    InitAtaeLstmParam(&initParam);
    auto lstmBase = std::make_shared<AtaeLstmBase>();
    APP_ERROR ret = lstmBase->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "LstmBase init failed, ret=" << ret << ".";
        return ret;
    }

    // get all file name
    std::string inferPath = argv[1];
    std::vector<std::string> files;
    ret = ReadFilesFromPath(inferPath + "/00_content", &files);
    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        return ret;
    }

    // do process
    for (uint32_t i = 0; i < files.size(); i++) {
        LogInfo << "read file name: " << files[i];
        ret = lstmBase->Process(inferPath, files[i]);
        if (ret != APP_ERR_OK) {
            LogError << "Lstmbase process failed, ret=" << ret << ".";
            lstmBase->DeInit();
            return ret;
        }
    }
    lstmBase->DeInit();

    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer sum: " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " bin/sec.";

    return APP_ERR_OK;
}

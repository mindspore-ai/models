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
#include "Relationnet.h"
#include "MxBase/Log/Log.h"


std::vector<double> g_inferCost;


void InitRelationnetParam(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->modelPath = "../data/model/relationnet.om";
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
    return APP_ERR_OK;
}


int main(int argc, char* argv[]) {
    InitParam initParam;
    InitRelationnetParam(&initParam);
    auto relationnet = std::make_shared<Relationnet>();
    APP_ERROR ret = relationnet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Relationnet init failed, ret=" << ret << ".";
        return ret;
    }
    std::string inferPath = "../data/input/";
    std::vector<std::string> files;
    ret = ReadFilesFromPath(inferPath, &files);
    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        return ret;
    }
    // do infer
    for (uint32_t i = 0; i < files.size(); i++) {
        ret = relationnet->Process(inferPath, files[i]);
        if (ret != APP_ERR_OK) {
            LogError << "Relationnet process failed, ret=" << ret << ".";
            relationnet->DeInit();
            return ret;
        }
    }
    LogInfo << "infer succeed and write the result data with binary file !";
    relationnet->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " bin/sec.";
    LogInfo << "\n == The infer result has been saved in ./result ==";
    return APP_ERR_OK;
}

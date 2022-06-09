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

#include <dirent.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include "SGCN.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void InitSgcnParam(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->modelPath = "../convert/sgcn.om";
}

int main(int argc, char* argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input dataset path and dataset type";
        return APP_ERR_OK;
    }

    InitParam initParam;
    InitSgcnParam(&initParam);
    auto sgcnBase = std::make_shared<sgcn>();
    APP_ERROR ret = sgcnBase->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FFMbase init failed, ret=" << ret << ".";
        return ret;
    }
    std::string inferPath = argv[1];
    std::string dataType = argv[2];
    std::vector<std::string> files;
    files.push_back(argv[1]);
    for (uint32_t i = 0; i < files.size(); i++) {
        LogInfo << "read file name: " << files[i];
        ret = sgcnBase->Process(inferPath, dataType);
        if (ret != APP_ERR_OK) {
            LogError << "Gcnbase process failed, ret=" << ret << ".";
            sgcnBase->DeInit();
            return ret;
        }
        LogInfo << "Finish " << i + 1 << " file";
    }
    LogInfo << "======== Inference finished ========";
    sgcnBase->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " bin/sec.";
    return APP_ERR_OK;
}


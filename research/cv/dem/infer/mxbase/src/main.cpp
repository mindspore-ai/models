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

#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "DEM.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void InitDEMParam(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->modelPath = "../convert/dem.om";
}

int main(int argc, char* argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input dataset and path e.g. ./dem [dataset] [data_path]";
        return APP_ERR_OK;
    }

    InitParam initParam;
    InitDEMParam(&initParam);
    auto demBase = std::make_shared<DEM>();
    APP_ERROR ret = demBase->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "DEMBase init failed, ret=" << ret << ".";
        return ret;
    }
    std::string dataset = argv[1];
    std::string inferPath = argv[2];
    uint32_t size = (dataset == "CUB") ? 312 : 85;
    LogInfo << "Infer path :" << inferPath;
    int len = (dataset == "CUB") ? 50 : 10;
    for (int i = 0; i < len; ++i) {
        char file[1024];
        snprintf(file, sizeof(file), "test_att_%d", i);
        LogInfo << "reading file name:" << file;
        ret = demBase->Process(inferPath + file, size);
        if (ret != APP_ERR_OK) {
            LogError << "DEMBase process failed, ret=" << ret << ".";
            demBase->DeInit();
            return ret;
        }
        LogInfo << "Finish " << i << " file";
    }
    LogInfo << "======== Inference finished ========";
    demBase->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " bin/sec.";
    return APP_ERR_OK;
}

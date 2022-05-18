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

#include "Hypertext.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void InitHypertextParam(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->modelType = "tnews";
    initParam->maxLength = 40;
    initParam->modelPath = "../data/model/tnews.om";
    initParam->inferIdsPath = "../data/input/tnews_infer_txt/hypertext_ids_bs1_57404.txt";
    initParam->inferNgradPath = "../data/input/tnews_infer_txt/hypertext_ngrad_bs1_57404.txt";
    initParam->resultName = "result_tnews.txt";
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        LogWarn << "Please input model_type, max_length, model_path, infer_id_path, infer_ngrad_path and result_name.";
        return APP_ERR_OK;
    }
    InitParam initParam;
    InitHypertextParam(&initParam);
    initParam.modelType = argv[1];
    initParam.maxLength = atoi(argv[2]);
    initParam.modelPath = argv[3];
    initParam.inferIdsPath = argv[4];
    initParam.inferNgradPath = argv[5];
    initParam.resultName = argv[6];
    auto hypertextBase = std::make_shared<HypertextNerBase>();
    APP_ERROR ret = hypertextBase->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Hypertextbase init failed, ret=" << ret << ".";
        return ret;
    }
    // process
    ret = hypertextBase->Process(initParam.inferIdsPath, initParam.inferNgradPath, initParam.modelType);
    if (ret != APP_ERR_OK) {
        LogError << "Hypertextbase process failed, ret=" << ret << ".";
        hypertextBase->DeInit();
        return ret;
    }
    hypertextBase->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer texts sum " << g_inferCost.size()
            << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum
            << " bin/sec.";
    return APP_ERR_OK;
}

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

#include "Fasttext.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void InitFasttextParam(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->modelPath = "../data/model/fasttext_agnews.om";
    initParam->inferSrcTokensPath = "../data/input/src_tokens.txt";
    initParam->resultName = "mxbase_predictions_sens.txt";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        LogWarn << "Please input model_path, infer_src_tokens_path and result_name.";
        return APP_ERR_OK;
    }
    InitParam initParam;
    InitFasttextParam(&initParam);
    initParam.modelPath = argv[1];
    initParam.inferSrcTokensPath = argv[2];
    initParam.resultName = argv[3];
    auto fasttextBase = std::make_shared<FasttextNerBase>();
    APP_ERROR ret = fasttextBase->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Fasttextbase init failed, ret=" << ret << ".";
        return ret;
    }
    // process
    ret = fasttextBase->Process(initParam.inferSrcTokensPath, initParam.resultName);
    if (ret != APP_ERR_OK) {
        LogError << "Fasttextbase process failed, ret=" << ret << ".";
        fasttextBase->DeInit();
        return ret;
    }
    fasttextBase->DeInit();
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

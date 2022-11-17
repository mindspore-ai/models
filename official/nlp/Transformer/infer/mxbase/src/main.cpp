
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

#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "Transformer.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void InitProtonetParam(InitParam* initParam, const std::string &model_path, const std::string &output_data_path) {
    initParam->deviceId = 0;
    initParam->modelPath = model_path;
    initParam->outputDataPath = output_data_path;
}

int main(int argc, char* argv[]) {
    LogInfo << "=======================================  !!!Parameters setting!!!" << \
               "========================================";
    std::string model_path = argv[1];
    LogInfo << "==========  loading model weights from: " << model_path;

    std::string input_data_path = argv[2];
    LogInfo << "==========  input data path = " << input_data_path;

    std::string output_data_path = argv[3];
    LogInfo << "==========  output data path = " << output_data_path << \
               " WARNING: please make sure that this folder is created in advance!!!";

    LogInfo << "========================================  !!!Parameters setting!!! " << \
               "========================================";

    InitParam initParam;
    InitProtonetParam(&initParam, model_path, output_data_path);
    auto model_ = std::make_shared<transformer>();
    APP_ERROR ret = model_->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "transformer init failed, ret=" << ret << ".";
        return ret;
    }
    for (uint32_t i = 0; i < 3003; i++) {
        ret = model_->Process(input_data_path, "transformer_bs_1_" + std::to_string(i) + ".bin");
        if (ret != APP_ERR_OK) {
            LogError << "transformer process failed, ret=" << ret << ".";
            model_->DeInit();
            return ret;
    }
    }
    LogInfo << "infer succeed and write the result data with binary file !";

    model_->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " bin/sec.";
    LogInfo << "==========  The infer result has been saved in ---> " << output_data_path;
    return APP_ERR_OK;
}

/*
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dirent.h>
#include <fstream>
#include "MxBase/Log/Log.h"
#include "Tbnet.h"

namespace {
    const uint32_t DATA_SIZE = 18415;
}  // namespace

int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/tbnet.om";
    std::string dataPath = "../../preprocess_Result/";

    auto model_Tbnet = std::make_shared<Tbnet>();
    APP_ERROR ret = model_Tbnet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Tagging init failed, ret=" << ret << ".";
        model_Tbnet->DeInit();
        return ret;
    }

    std::vector<int> outputs;
    for (int i=0; i < DATA_SIZE; i++) {
        LogInfo << "processing " << i;
        ret = model_Tbnet->Process(i, dataPath, initParam, outputs);
        if (ret !=APP_ERR_OK) {
            LogError << "Tbnet process failed, ret=" << ret << ".";
            model_Tbnet->DeInit();
            return ret;
        }
    }

    model_Tbnet->DeInit();

    double total_time = model_Tbnet->GetInferCostMilliSec() / 1000;
    LogInfo<< "inferance total cost time: "<< total_time<< ", FPS: "<< DATA_SIZE/total_time;

    return APP_ERR_OK;
}

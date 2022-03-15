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

#include "Pgan.h"

namespace {
    const float MS_TO_S = 1000.0;
}

int main(int argc, char* argv[]) {
    std::string modelPath = argv[1];  // .om
    std::string srPath = argv[2];  // output path
    std::string batchSize = argv[3];
    uint32_t batch = strtoul(batchSize.c_str(), NULL, 10);

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = modelPath;
    initParam.srPath = srPath;

    // init the model
    auto pgan = std::make_shared<Pgan>();
    APP_ERROR ret = pgan->Init(initParam, batch);
    if (ret != APP_ERR_OK) {
        LogError << "Pgan init failed, ret=" << ret << ".";
        return ret;
    }

    // start to record the process time of the model.
    auto startTime = std::chrono::high_resolution_clock::now();

    ret = pgan->Process(batch);

    if (ret != APP_ERR_OK) {
        LogError << "Pgan process failed, ret=" << ret << ".";
        pgan->DeInit();
        return ret;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    pgan->DeInit();

    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = MS_TO_S * batch/ pgan->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}

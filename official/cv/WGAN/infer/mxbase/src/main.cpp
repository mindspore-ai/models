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
#include <iostream>
#include <fstream>
#include <vector>
#include "WGAN.h"
#include "MxBase/Log/Log.h"

void init_Param(InitParam *initParam, model_info *modelInfo) {
    initParam->deviceId = 0;
    initParam->modelPath = "../data/model/DCGAN/WGAN.om";

    modelInfo->noise_length = 100;
    modelInfo->nimages = 1;
    modelInfo->image_size = 64;
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input output result path, such as '../data/mxbase_result/'.";
        return APP_ERR_OK;
    }
    InitParam initParam;
    model_info modelInfo;
    init_Param(&initParam, &modelInfo);

    auto wgan = std::make_shared<WGAN>();
    APP_ERROR ret = wgan->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "WGAN init failed, ret=" << ret << ".";
        return ret;
    }

    std::string resultPath = argv[1];
    ret = wgan->Process(modelInfo, resultPath);
    if (ret != APP_ERR_OK) {
        LogError << "WGAN process failed, ret=" << ret << ".";
        wgan->DeInit();
        return ret;
    }

    wgan->DeInit();
    return APP_ERR_OK;
}

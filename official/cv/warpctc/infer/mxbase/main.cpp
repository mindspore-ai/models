/*
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

#include <warpctc.h>
#include <iostream>
#include <vector>
#include "MxBase/Log/Log.h"


void InitwarpCTCParam(InitParam &initParam) {
    initParam.deviceId = 0;
    initParam.resize_h = 64;
    initParam.resize_w = 160;
    initParam.modelPath = "../convert/warpctc_3.om";
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path.";
        return APP_ERR_OK;
    }

    InitParam initParam;
    InitwarpCTCParam(initParam);
    auto warpctc = std::make_shared<warpCTC>();
    LogInfo << "model path: " << initParam.modelPath;
    LogInfo << "resize_h: " << initParam.resize_h;
    LogInfo << "resize_w: " << initParam.resize_w;
    APP_ERROR ret = warpctc->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "warpCTC init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    ret = warpctc->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "warpCTC process failed, ret=" << ret << ".";
        warpctc->DeInit();
        return ret;
    }
    warpctc->DeInit();
    return APP_ERR_OK;
}



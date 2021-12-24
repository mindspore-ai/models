/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "src/hrnet.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NUM = 19;
    const uint32_t MODEL_TYPE = 1;
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './build/hrnetw48seg test.png'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.modelType = MODEL_TYPE;
    initParam.labelPath = "../data/config/hrnetw48seg.names";
    initParam.modelPath = "../data/model/hrnetw48seg.om";
    HRNetW48Seg hrnetw48seg;
    APP_ERROR ret = hrnetw48seg.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "HRNetW48Seg init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgPath = argv[1];
    ret = hrnetw48seg.Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "HRNetW48Seg process failed, ret=" << ret << ".";
        hrnetw48seg.DeInit();
        return ret;
    }
    hrnetw48seg.DeInit();

    return APP_ERR_OK;
}

/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "FCN8s.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NUM = 21;
    const uint32_t MODEL_TYPE = 1;
    const uint32_t FRAMEWORK_TYPE = 2;
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './fcn8s test.jpg'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.modelType = MODEL_TYPE;
    initParam.labelPath = "../data/config/FCN8s.names";
    initParam.modelPath = "../data/model/FCN8s.om";
    initParam.checkModel = true;
    initParam.frameworkType = FRAMEWORK_TYPE;

    FCN8s fcn8s;
    APP_ERROR ret = fcn8s.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FCN8s init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    ret = fcn8s.Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "FCN8s process failed, ret=" << ret << ".";
        fcn8s.DeInit();
        return ret;
    }
    fcn8s.DeInit();
    return APP_ERR_OK;
}

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
#include "FQA.h"
#include "MxBase/Log/Log.h"

int main(int argc, char *argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input test dataset path";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = argv[2];
    FQA fqa;
    APP_ERROR ret = fqa.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FQA init failed, ret=" << ret << ".";
        return ret;
    }

    std::string testPath = argv[1];
    ret = fqa.Process(testPath);
    if (ret != APP_ERR_OK) {
        LogError << "FQA process failed, ret=" << ret << ".";
        fqa.DeInit();
        return ret;
    }

    fqa.DeInit();
    return APP_ERR_OK;
}

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

#include "RcanSuperresolution.h"
#include "MxBase/Log/Log.h"


// infer an image
int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './test.png'";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../model/rcan.om";
    RcanSuperresolution rcanSR;
    APP_ERROR ret = rcanSR.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "RcanSuperresolution init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgPath = argv[1];
    ret = rcanSR.Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "RcanSuperresolution process failed, ret=" << ret << ".";
        rcanSR.DeInit();
        return ret;
    }

    rcanSR.DeInit();
    return APP_ERR_OK;
}

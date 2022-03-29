/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd.
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
#include <vector>
#include <FaceboxesDetection.h>
#include "MxBase/Log/Log.h"


void InitFaceboxesParam(InitParam &initParam) {
    initParam.deviceId = 0;
    initParam.modelPath = "../../data/model/FaceBoxes.om";
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './FaceboxesPostProcess test.jpg'.";
        return APP_ERR_OK;
    }

    InitParam initParam;
    InitFaceboxesParam(initParam);
    auto facebox = std::make_shared<FaceboxesDetection>();
    APP_ERROR ret = facebox->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FaceboxesDetection init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    ret = facebox->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "FaceboxesDetection process failed, ret=" << ret << ".";
        facebox->DeInit();
        return ret;
    }
    facebox->DeInit();
    return APP_ERR_OK;
}

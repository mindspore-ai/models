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

#include "SimplePOSE.h"
#include "MxBase/Log/Log.h"

namespace {
const uint32_t DEVICE_ID = 0;
const char *RESULT_PATH = "./infer_results/";
const char *BBOX_FILE = "../sdk/files/COCO_val2017_detections_AP_H_56_person.json";
}  // namespace

int main(int argc, char *argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path, such as './SimplePOSE_mindspore [om_file_path] [img_path]'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = DEVICE_ID;
    initParam.checkTensor = true;
    initParam.modelPath = argv[1];
    auto inferSimplePOSE = std::make_shared<SimplePOSE>();
    APP_ERROR ret = inferSimplePOSE->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "SimplePOSE init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[2];
    ret = inferSimplePOSE->Process(BBOX_FILE, imgPath, RESULT_PATH);
    if (ret != APP_ERR_OK) {
        LogError << "SimplePOSE process failed, ret=" << ret << ".";
        inferSimplePOSE->DeInit();
        return ret;
    }
    inferSimplePOSE->DeInit();
    return APP_ERR_OK;
}

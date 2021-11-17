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
#include "CenterNet.h"
#include "MxBase/Log/Log.h"

namespace {
const uint32_t DEVICE_ID = 0;
const char RESULT_PATH[] = "../data/predict_result.json";

// parameters of post process
const uint32_t CLASS_NUM = 80;
const char LABEL_PATH[] = "../data/config/coco2017.names";

}  // namespace

int main(int argc, char *argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path, such as './centernet_mindspore [om_file_path] [img_path]'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = DEVICE_ID;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = LABEL_PATH;
    initParam.checkTensor = true;
    initParam.modelPath = argv[1];
    auto inferCenterNet = std::make_shared<CenterNet>();
    APP_ERROR ret = inferCenterNet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "CenterNet init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[2];
    ret = inferCenterNet->Process(imgPath, RESULT_PATH);
    if (ret != APP_ERR_OK) {
        LogError << "CenterNet process failed, ret=" << ret << ".";
        inferCenterNet->DeInit();
        return ret;
    }
    inferCenterNet->DeInit();
    return APP_ERR_OK;
}

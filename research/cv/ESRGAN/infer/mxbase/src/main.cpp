/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


#include "MxBase/Log/Log.h"
#include "ESRGAN.h"

namespace {
const uint32_t DEVICE_ID = 0;
}  // namespace

int main(int argc, char *argv[]) {
    if (argc <= 3) {
        LogWarn << "Please input image path, such as './ [om_file_path] [img_path] [result_path]'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = DEVICE_ID;


    initParam.checkTensor = true;

    initParam.modelPath = argv[1];
    auto inferESRGAN = std::make_shared<ESRGAN>();
    APP_ERROR ret = inferESRGAN->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "ESRGAN init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgPath = argv[2];
    std::string result_path = argv[3];
    ret = inferESRGAN->Process(imgPath, result_path);
    if (ret != APP_ERR_OK) {
        LogError << "ESRGAN process failed, ret=" << ret << ".";
        inferESRGAN->DeInit();
        return ret;
    }
    inferESRGAN->DeInit();
    return APP_ERR_OK;
}

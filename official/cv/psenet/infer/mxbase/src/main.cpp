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

#include <experimental/filesystem>
#include <string>
#include <vector>
#include "PsenetDetection.h"
#include "MxBase/Log/Log.h"
namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.kernelNum = 7;
    initParam.pseScale = 1.0;
    initParam.minKernelArea = 5.0;
    initParam.minScore = 0.93;
    initParam.minArea = 800.0;
    initParam.labelPath = "imagenet1000_clsidx_to_labels.names";
    initParam.checkTensor = true;
    initParam.modelPath = "../convert/psenet.om";
    auto psenet = std::make_shared<PsenetDetection>();
    APP_ERROR ret = psenet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "psenet init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgDir = "../data/image/ch4_test_images";
    for (auto & entry : fs::directory_iterator(imgDir)) {
        LogInfo << "read image path" << entry.path();
        ret = psenet->Process(entry.path());
        if (ret != APP_ERR_OK) {
            LogError << "psenet process failed, ret=" << ret << ".";
            psenet->DeInit();
            return ret;
        }
    }
    psenet->DeInit();
    return APP_ERR_OK;
}

/*
 * Copyright (c) 2022 Huawei Technologies Co., Ltd. All rights reserved.
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

#include <dirent.h>
#include <unistd.h>
#include<fstream>
#include<string>
#include "MetricLearn.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t DEVICE_ID = 0;
    const char RESULT_PATH[] = "../data/preds/mxbase";
    const char MODEL_PATH[] = "../convert/resnet50_acc74_aippnorm.om";
}  // namespace


int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './metric_learn image_dir'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = DEVICE_ID;
    initParam.modelPath = MODEL_PATH;
    auto metric_learn = std::make_shared<MetricLearn>();
    APP_ERROR ret = metric_learn->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "MetricLearn init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    std::vector<std::string> imgFilePaths;

    // read test_half.txt
    DIR *dirPtr = opendir(imgPath.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << imgPath;
        return APP_ERR_INTERNAL_ERROR;
    }

    std::fstream f(imgPath + "/test_half.txt");
    std::string line;
    while (getline(f, line)) {
        int count = 0;
        std::string filePath;
        for (std::size_t i = 0; i < line.size(); i++) {
            count++;
            if (line[i] == ' ') {
                filePath = line.substr(i - count + 1, count - 1);
            }
        }
        imgFilePaths.emplace_back(imgPath + "/" + filePath);
    }
    f.close();

    auto startTime = std::chrono::high_resolution_clock::now();
    for (auto &imgFile : imgFilePaths) {
        ret = metric_learn->Process(imgFile, RESULT_PATH);
        if (ret != APP_ERR_OK) {
            LogError << "MetricLearn process failed, ret=" << ret << ".";
            metric_learn->DeInit();
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    metric_learn->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgFilePaths.size() / metric_learn->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}

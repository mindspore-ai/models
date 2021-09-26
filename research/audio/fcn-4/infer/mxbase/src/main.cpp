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

#include <dirent.h>
#include "MxBase/Log/Log.h"
#include "FCN-4.h"

namespace {
    const uint32_t CLASS_NUM = 50;
    const uint32_t BATCH_SIZE = 1;
}  // namespace

APP_ERROR ScanMelgramFiles(const std::string &rootPath, std::vector<std::string> *featureFiles) {
    DIR *dirPtr = opendir(rootPath.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << rootPath;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }
        featureFiles->push_back(rootPath + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input feature file store path, such as '../../data/melgram'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../../data/config/tag.txt";
    initParam.topk = 50;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = "../../data/models/fcn-4.om";
    auto fcn_4 = std::make_shared<FCN_4>();
    APP_ERROR ret = fcn_4->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FCN-4 Tagging init failed, ret=" << ret << ".";
        return ret;
    }
    std::string rootPath = argv[1];
    std::vector<std::string> melFeaturePaths;
    ret = ScanMelgramFiles(rootPath, &melFeaturePaths);
    if (ret != APP_ERR_OK) {
        fcn_4->DeInit();
        return ret;
    }
    int sampleCount = 0;
    LogInfo << "Number of total feature files load from input data path: " << melFeaturePaths.size();
    for (uint32_t i = 0; i <= melFeaturePaths.size() - BATCH_SIZE; i += BATCH_SIZE) {
        std::vector<std::string> batchMelFeaturePaths(melFeaturePaths.begin()+i,
                                                        melFeaturePaths.begin()+(i+BATCH_SIZE));
        ret = fcn_4->Process(batchMelFeaturePaths);
        if (ret != APP_ERR_OK) {
            LogError << "FCN-4 Tagging process failed, ret=" << ret << ".";
            fcn_4->DeInit();
            return ret;
        }
        sampleCount += BATCH_SIZE;
    }

    fcn_4->DeInit();
    double fps = 1000.0 * sampleCount / fcn_4->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << fcn_4->GetInferCostMilliSec() << " ms\tfps: " << fps << " audio_clips/sec";
    return APP_ERR_OK;
}

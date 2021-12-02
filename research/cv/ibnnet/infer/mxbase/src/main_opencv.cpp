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

#include <dirent.h>
#include "IbnnetOpencv.h"
#include "MxBase/Log/Log.h"

namespace {
const uint32_t DEVICE_ID = 0;
const uint32_t CLASS_NUM = 1000;
const char RESULT_PATH[] = "./preds/";
const uint32_t TOP_K = 5;
const char MODEL_PATH[] = "../data/model/ibnnet.om";
const char LABEL_PATH[] = "../data/config/imagenet1000_clsidx_to_labels.names";
}  // namespace


int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './ibnnet image_dir'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = DEVICE_ID;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = LABEL_PATH;
    initParam.topk = TOP_K;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = MODEL_PATH;
    auto ibnnet = std::make_shared<IbnnetClassifyOpencv>();
    APP_ERROR ret = ibnnet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "IbnnetClassifyDvpp init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    std::vector<std::string> imgFilePaths;

    // scan all images
    DIR *dirPtr = opendir(imgPath.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << imgPath;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        imgFilePaths.emplace_back(imgPath + "/" + fileName);
    }
    closedir(dirPtr);

    auto startTime = std::chrono::high_resolution_clock::now();
    for (auto &imgFile : imgFilePaths) {
        ret = ibnnet->Process(imgFile, RESULT_PATH);
        if (ret != APP_ERR_OK) {
            LogError << "IbnnetClassifyDvpp process failed, ret=" << ret << ".";
            ibnnet->DeInit();
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    ibnnet->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgFilePaths.size() / ibnnet->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}

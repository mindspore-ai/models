 /*
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "SPNASNET.h"

namespace {
    const uint32_t CLASS_NUM = 1000;
    const uint32_t BATCH_SIZE = 1;
}  // namespace

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> *imgFiles) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        imgFiles->push_back(path + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as '../../data/images'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../../imagenet1000_clsidx_to_labels.names";
    initParam.topk = 5;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = "../../spnasnet.om";
    auto spnasnet = std::make_shared<SPNASNET>();
    APP_ERROR ret = spnasnet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "spnasnetClassify init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    std::vector<std::string> imgFilePaths;
    ret = ScanImages(imgPath, &imgFilePaths);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    int inferImgsCount = 0;
    LogInfo << "Number of total images load from input data path: " << imgFilePaths.size();
    for (uint32_t i = 0; i <= imgFilePaths.size() - BATCH_SIZE; i += BATCH_SIZE) {
        std::vector<std::string>batchImgFilePaths(imgFilePaths.begin()+i, imgFilePaths.begin()+(i+BATCH_SIZE));
        ret = spnasnet->Process(batchImgFilePaths);
        if (ret != APP_ERR_OK) {
            LogError << "SPNASNETClassify process failed, ret=" << ret << ".";
            spnasnet->DeInit();
            return ret;
        }
        inferImgsCount += BATCH_SIZE;
    }

    spnasnet->DeInit();
    double fps = 1000.0 * inferImgsCount / spnasnet->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << spnasnet->GetInferCostMilliSec() << " ms\tfps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}
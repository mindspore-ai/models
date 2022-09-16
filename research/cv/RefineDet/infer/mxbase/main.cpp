/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "refinedetDetection/refinedetDetection.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_infer_cost;

namespace {
    const uint32_t CLASS_NU = 81;
    const uint32_t BOX_DIM = 4;
    const uint32_t RESIZE_WIDTH = 320;
    const uint32_t RESIZE_HEIGHT = 320;

    const uint32_t MAX_BOXES = 100;
    const uint32_t NMS_THERSHOLD = 0.6;
    const uint32_t MIN_SCORE = 0.1;
    const uint32_t NUM_RETINANETBOXES = 6375;
}   // namespace

static APP_ERROR init_refinedet_param(InitParam *initParam) {
    initParam->deviceId = 0;
    initParam->labelPath = ".../data/config/coco2017.names";
    initParam->modelPath = ".../data/model/refinedet.om";
    initParam->resizeWidth = RESIZE_WIDTH;
    initParam->resizeHeight = RESIZE_HEIGHT;
    initParam->width = 0;
    initParam->height = 0;
    initParam->maxBoxes = MAX_BOXES;
    initParam->nmsThershold = NMS_THERSHOLD;
    initParam->minScore = MIN_SCORE;
    initParam->numRetinanetBoxes = NUM_RETINANETBOXES;
    initParam->classNum = CLASS_NU;
    initParam->boxDim = BOX_DIM;

    return APP_ERR_OK;
}

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> &imgFiles) {
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

        imgFiles.emplace_back(path + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './refinedet img_dir'.";
        return APP_ERR_OK;
    }
    InitParam initParam;
    int ret = init_refinedet_param(&initParam);
    if (ret != APP_ERR_OK) {
        LogError << "InitrefinedetParam Init failed, ret=" << ret << ".";
        return ret;
    }
    auto refinedet = std::make_shared<refinedetDetection>();
    ret = refinedet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "refinedetDetection Init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgPath = argv[1];
    std::vector<std::string> imgFilePaths;
    ret = ScanImages(imgPath, imgFilePaths);
    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        return ret;
    }
    for (auto &imgName : imgFilePaths) {
        ret = refinedet->process(imgName, initParam);
        if (ret != APP_ERR_OK) {
            LogError << "refinedetDetection process failed, ret=" << ret << ".";
            refinedet->DeInit();
            return ret;
         }
    }
    refinedet->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_infer_cost.size(); i++) {
        costSum += g_infer_cost[i];
    }
    LogInfo << "Infer images sum " << g_infer_cost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_infer_cost.size() * 1000 / costSum << " bin/sec.";
    return APP_ERR_OK;
}

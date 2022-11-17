/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "retinanetDetection/RetinanetDetection.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_infer_cost;

namespace {
    const uint32_t CLASS_NU = 81;
    const uint32_t BOX_DIM = 4;
    const uint32_t RESIZE_WIDTH = 600;
    const uint32_t RESIZE_HEIGHT = 600;

    const uint32_t MAX_BOXES = 100;
    const uint32_t NMS_THERSHOLD = 0.6;
    const uint32_t MIN_SCORE = 0.1;
    const uint32_t NUM_RETINANETBOXES = 67995;
}   // namespace

static APP_ERROR init_retinanet_param(InitParam *initParam) {
    initParam->deviceId = 0;
    initParam->labelPath = "./model/coco.names";
    initParam->modelPath = "./model/retinanet.om";
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

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './retinanet test.jpg'.";
        return APP_ERR_OK;
    }
    InitParam initParam;
    int ret = init_retinanet_param(&initParam);
    if (ret != APP_ERR_OK) {
        LogError << "InitRetinanetParam Init failed, ret=" << ret << ".";
        return ret;
    }
    auto retinanet = std::make_shared<RetinanetDetection>();
    ret = retinanet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "RetinanetDetection Init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgName = argv[1];
    ret = retinanet->process(imgName, initParam);
    if (ret != APP_ERR_OK) {
        LogError << "RetinaneDetection process failed, ret=" << ret << ".";
        retinanet->DeInit();
        return ret;
    }
    retinanet->DeInit();

    double costSum = 0;
    for (uint32_t i = 0; i < g_infer_cost.size(); i++) {
        costSum += g_infer_cost[i];
    }
    LogInfo << "Infer images sum " << g_infer_cost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_infer_cost.size() * 1000 / costSum << " bin/sec.";
    return APP_ERR_OK;
}

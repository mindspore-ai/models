/**
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

// c++的入口文件
#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "SSDResNet50.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NUM = 81;
}

int main(int argc, char *argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path, such as './ssd_resnet50 ssd_resnet50.om test.jpg'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../models/coco.names";

    initParam.iou_thresh = 0.6;
    initParam.score_thresh = 0.6;
    initParam.checkTensor = true;

    initParam.modelPath = argv[1];
    auto ssdResnet50 = std::make_shared<SSDResNet50>();
    APP_ERROR ret = ssdResnet50->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "SsdResnet50 init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[2];
    ret = ssdResnet50->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "SsdResnet50 process failed, ret=" << ret << ".";
        ssdResnet50->DeInit();
        return ret;
    }
    ssdResnet50->DeInit();
    return APP_ERR_OK;
}

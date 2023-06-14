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
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>

#include "FaceRecognition.h"
#include "MxBase/Log/Log.h"

namespace {
const int emb_size = 256;
}  // namespace



APP_ERROR ScanImages(const std::string &path) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}


int main(int argc, char* argv[]) {
    if (argc <= 5) {
        LogWarn << "Please input image path, such as './lists image_dir'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = emb_size;
    initParam.labelPath = argv[1];
    initParam.topk = emb_size;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = argv[2];
    auto resnet50 = std::make_shared<FaceRecognition>();
    APP_ERROR ret = resnet50->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "facerecognition init failed, ret=" << ret << ".";
        return ret;
    }

    std::string zj_list_path = argv[3];
    std::string jk_list_path = argv[4];
    std::string dis_list_path = argv[5];

    ret = resnet50->main(zj_list_path, jk_list_path, dis_list_path);
    if (ret != APP_ERR_OK) {
        LogError << "facerecognition main failed, ret=" << ret << ".";
        resnet50->DeInit();
        return ret;
    }
    resnet50->DeInit();
    return APP_ERR_OK;
}


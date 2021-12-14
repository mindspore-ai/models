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

#include <dirent.h>
#include <experimental/filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "CnnctcDetection.h"
#include "MxBase/Log/Log.h"
namespace fs = std::experimental::filesystem;

namespace CONST_CONFIG {
const uint32_t CLASS_NUM = 37;
const uint32_t OBJECT_NUM = 26;
const uint32_t BLANK_INDEX = 36;
}

int fileNameFilter(const struct dirent *cur) {
    std::string str(cur->d_name);
    if (str.find(".jpg") != std::string::npos) {
        return 1;
    }
    return 0;
}

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> *imgFiles) {
    struct dirent **namelist;
    int n = scandir(path.c_str(), &namelist, fileNameFilter, alphasort);
    if (n < 0) {
        LogError << "opendir failed. dir: " << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    for (int i = 0; i < n; ++i) {
        std::string filePath(namelist[i]->d_name);
        imgFiles->push_back(path + "/" + filePath);
        free(namelist[i]);
    }
    free(namelist);
    return APP_ERR_OK;
}

void SetInitParam(InitParam *initParam) {
    initParam->deviceId = 0;
    initParam->classNum = CONST_CONFIG::CLASS_NUM;
    initParam->objectNum = CONST_CONFIG::OBJECT_NUM;
    initParam->blankIndex = CONST_CONFIG::BLANK_INDEX;
    initParam->labelPath = "../../sdk/crnn.names";
    initParam->argmax = false;
    initParam->modelPath = "../../data/model/cnnctc.om";
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './xception image_dir'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    SetInitParam(&initParam);

    auto cnnctc = std::make_shared<CnnctcDetection>();
    APP_ERROR ret = cnnctc->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "cnnctc init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgDir = argv[1];
    // 检测result文件夹是否存在，不存在需要创建该文件夹
    std::ofstream infer_ret("../../result/cnnctc_mxbase_result.txt");
    std::vector<std::string> imgFilePaths;
    ScanImages(imgDir, &imgFilePaths);
    for (auto &imgFile : imgFilePaths) {
        std::string result = "";
        ret = cnnctc->Process(imgFile, &result);
        infer_ret << result << std::endl;
        if (ret != APP_ERR_OK) {
            LogError << "cnnctc process failed, ret=" << ret << ".";
            cnnctc->DeInit();
            return ret;
        }
    }
    cnnctc->DeInit();
    infer_ret.close();
    return APP_ERR_OK;
}

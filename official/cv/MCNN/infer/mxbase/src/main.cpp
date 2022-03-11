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
 * ============================================================================
 */

#include "Mcnn.h"


APP_ERROR ScanImages(const std::string &path, std::vector<std::string> *imgFiles) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..")
            continue;

        imgFiles->emplace_back(fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}


int main(int argc, char* argv[]) {
    if (argc <= 4) {
        LogWarn << "Please input image path, such as './Mcnn [model_path] [data_path] [label_path] [output_path]'.";
        return APP_ERR_OK;
    }

    const std::string modelPath = argv[1];
    std::string inputPath = argv[2];
    std::string gtPath = argv[3];
    std::string srPath = argv[4];

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = modelPath;
    initParam.srPath = srPath;
    initParam.gtPath = gtPath;

    auto mcnn = std::make_shared<Mcnn>();
    APP_ERROR ret = mcnn->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "mcnn init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::string> imgFilePaths;
    ret = ScanImages(inputPath, &imgFilePaths);
    if (ret != APP_ERR_OK) {
        LogError << "mcnn lq img scan error, ret=" << ret << ".";
        return ret;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    int totalNum = 0;

    sort(imgFilePaths.begin(), imgFilePaths.end());
    int imgFilePaths_size = imgFilePaths.size();
    for (int i = 0; i < imgFilePaths_size; i++) {
        LogInfo << imgFilePaths[i];
    }

    for (auto &imgFile : imgFilePaths) {
        LogInfo << totalNum;
        ret = mcnn->Process(inputPath+'/'+imgFile, imgFile);
        ++totalNum;
        if (ret != APP_ERR_OK) {
            LogError << "mcnn process failed, ret=" << ret << ".";
            mcnn->DeInit();
            return ret;
        }
    }
    float mae = mcnn->getmae()/totalNum;
    float mse = sqrt(mcnn->getmse()/totalNum);
    LogInfo << "mae:" << mae;
    LogInfo << "mse:" << mse;
    auto endTime = std::chrono::high_resolution_clock::now();
    mcnn->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgFilePaths.size() / mcnn->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}

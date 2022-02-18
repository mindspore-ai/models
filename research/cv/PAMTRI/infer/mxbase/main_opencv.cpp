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
#include <fstream>
#include "MxBase/Log/Log.h"
#include "PAMTRIClassifyOpencv.h"


APP_ERROR scan_images(const std::string &path,
                     std::vector<std::vector<std::string>> *imgFiles,
                     int batch_size) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    int bs = 0;
    std::vector<std::string> batchImg;
    std::set<std::string> temp_img;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }
        temp_img.insert(fileName);
    }
    for (auto fileName : temp_img) {
        if (fileName == "." || fileName == "..") {
            continue;
        }
        if (bs++ < batch_size) {
            batchImg.emplace_back(path + "/" + fileName);
        } else {
            imgFiles->emplace_back(batchImg);
            batchImg = std::vector<std::string>({path + "/" + fileName});
            bs = 1;
        }
    }
    imgFiles->emplace_back(batchImg);
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        LogWarn
            << "Please input image path and result path, such as ../data/veri ./result.txt";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.PoseEstNetPath = "../models/PoseEstNet.om";
    initParam.MultiTaskNetPath = "../models/MultiTask.om";
    initParam.resultPath = argv[2];
    initParam.segmentAware = true;
    initParam.heatmapAware = false;
    initParam.batchSize = 1;
    auto PAMTRI = std::make_shared<PAMTRIClassifyOpencv>(PAMTRIClassifyOpencv(initParam));
    APP_ERROR ret = PAMTRI->init_(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "PAMTRIClassify init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    std::vector<std::vector<std::string>> imgFilePaths;
    ret = scan_images(imgPath, &imgFilePaths, initParam.batchSize);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    for (auto &imgFile : imgFilePaths) {
        ret = PAMTRI->process_(imgFile);
        if (ret != APP_ERR_OK) {
            LogError << "PAMTRIClassify process failed, ret=" << ret << ".";
            PAMTRI->deinit_();
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    PAMTRI->deinit_();
    double costMilliSecs =
        std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgFilePaths.size() / costMilliSecs;
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps
            << " imgs/sec";
    return APP_ERR_OK;
}

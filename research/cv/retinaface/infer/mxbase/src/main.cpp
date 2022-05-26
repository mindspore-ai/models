/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd.
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

#include "RetinaFaceOpencv.h"
#include <dirent.h>
#include <unistd.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include "MxBase/Log/Log.h"

APP_ERROR ReadFilesFromPath(const std::string& path,
std::vector<std::string>* files) {
    DIR* dirPtr;
    // If it cannot be opened, skip
    if ((dirPtr = opendir(path.c_str())) == 0) {
        LogError << "opendir failed. dir:" << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    struct dirent* direntPtr = nullptr;
    std::string totalpath;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        // Check whether the file is a directory, if so, call this function recursively
        if (direntPtr->d_type == 4) {
            std::string subpathname = direntPtr->d_name;
            // Get rid of all of the current directory., the upper directory..
            if (subpathname == "." || subpathname == "..") {
                continue;
            }
            totalpath = path + subpathname;
            ReadFilesFromPath(totalpath, files);
        } else {
        // If not, save the file name of the bin traversed
            std::string fileName = direntPtr->d_name;
            // Get rid of all of the current directory., the upper directory..
            if (fileName == "." || fileName == "..") {
                continue;
            }
            // when get into else, path has changed
            int flag = path.rfind("/");
            std::string subpath = path.substr(flag + 1, path.length());
            files->push_back(subpath + "/" + fileName);
        }
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    // Init Param
    if (argc <= 1) {
        LogWarn << "Please input bin-file path, such as './retinaface bin_dir'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/Retinaface-huawei.om";
    std::string binPath = argv[1];
    std::string resultPath = "./mxbase_out/";
    auto retinaface = std::make_shared<RetinaFaceOpencv>();
    APP_ERROR ret = retinaface->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "RetinaFace init failed, ret=" << ret << ".";
        return ret;
    }

    // read bin files
    std::vector<std::string> binFiles;
    ret = ReadFilesFromPath(binPath, &binFiles);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    int cnt = 0;

    // if resultHomePath do not exists, create it
    if (access(resultPath.c_str(), 0) == -1) {
        std::string command = "mkdir -p " + resultPath;
        LogError << command.c_str() << std::endl;
        system(command.c_str());
    }

    // if the resultSubPath do not exists, create it
    for (uint32_t i = 0; i < binFiles.size(); i++) {
        int flag = binFiles[i].find("/");
        std::string subPath = resultPath + binFiles[i].substr(0, flag);
        if (access(subPath.c_str(), 0) == -1) {
            std::string command = "mkdir -p " + subPath;
            LogError << command.c_str() << std::endl;
            system(command.c_str());
        }
    }

    // do infer
    for (uint32_t i = 0; i < binFiles.size(); i++) {
        LogInfo << "Processing: " + std::to_string(i + 1) + "/" +
            std::to_string(binFiles.size()) + " ---> " + binFiles[i];
        ret = retinaface->Process(binPath, binFiles[i]);
        if (ret != APP_ERR_OK) {
            LogError << "Retinaface process failed, ret=" << ret << ".";
            retinaface->DeInit();
            return ret;
        }
        if (cnt++ % 1000 == 0) {
            LogError << cnt << std::endl;
        }
    }

    // get infer time
    auto endTime = std::chrono::high_resolution_clock::now();
    retinaface->DeInit();
    double costMilliSecs =
        std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * binPath.size() / retinaface->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps
        << " imgs/sec";
    return APP_ERR_OK;
}

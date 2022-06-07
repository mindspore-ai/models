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

#include "DepthNetOpencv.h"
#include <dirent.h>
#include <unistd.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
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
            // when get into else{}, path has changed
            int flag = path.rfind("/");
            std::string subpath = path.substr(flag + 1, path.length());
            if (fileName.substr(6, 6) == "colors")
                files->push_back(subpath + "/" + fileName);
            if (fileName.substr(6, 6) == "coarse")
                files->push_back(subpath + "/" + fileName);
        }
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    // Init Param
    if (argc <= 1) {
        LogWarn << "Please input bin-file path, such as './depthnet bin_dir'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.CoarseModelPath = "../input/FinalCoarseNet.om";
    initParam.FineModelPath = "../input/FinalFineNet.om";
    std::string RgbBinPath = argv[1];
    std::string CoarseDepthPath = "./coarse_infer_result";
    std::string FineDepthPath = "./fine_infer_result";
    auto depthnet = std::make_shared<DepthNetOpencv>();
    APP_ERROR ret = depthnet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "DepthNetOpencv init failed, ret=" << ret << ".";
        return ret;
    }

    // if CoarseDepthPath does not exist, create it
    if (access(CoarseDepthPath.c_str(), 0) == -1) {
        std::string command = "mkdir -p " + CoarseDepthPath;
        LogError << command.c_str() << std::endl;
        system(command.c_str());
    }

    // if FineDepthPath does not exist, create it
    if (access(FineDepthPath.c_str(), 0) == -1) {
        std::string command = "mkdir -p " + FineDepthPath;
        LogError << command.c_str() << std::endl;
        system(command.c_str());
    }

    // read rgb bin_files
    std::vector<std::string> RgbBinFiles;
    ret = ReadFilesFromPath(RgbBinPath, &RgbBinFiles);
    if (ret != APP_ERR_OK) {
        LogError << "Read rgbBinFiles failed, ret=" << ret << ".";
        return ret;
    }

    // Ensure that the Rgb files are arranged in order to correspond one to one, same as sdk
    std::sort(RgbBinFiles.begin(), RgbBinFiles.end());

    auto startTime = std::chrono::high_resolution_clock::now();
    int cnt = 0;

    // do coarse_infer
    for (uint32_t i = 0; i < RgbBinFiles.size(); i++) {
        LogInfo << "Coarse_Processing: " + std::to_string(i + 1) + "/" +
            std::to_string(RgbBinFiles.size()) + " ---> " + RgbBinFiles[i];
        ret = depthnet->CoarseProcess(RgbBinPath, RgbBinFiles[i]);
        if (ret != APP_ERR_OK) {
            LogError << "DepthNet coarse_process failed, ret=" << ret << ".";
            depthnet->DeInit();
            return ret;
        }
        if (cnt++ % 1000 == 0) {
            LogError << cnt << std::endl;
        }
    }

    // read coarse_infer_result_bin_files
    std::vector<std::string> CoarseDepthBinFiles;
    ret = ReadFilesFromPath(CoarseDepthPath, &CoarseDepthBinFiles);
    if (ret != APP_ERR_OK) {
        LogError << "Read CoarseDepthBinFiles failed, ret=" << ret << ".";
        return ret;
    }

    // Ensure that the Depth files are arranged in order to correspond one to one, same as sdk
    std::sort(CoarseDepthBinFiles.begin(), CoarseDepthBinFiles.end());

    // do fine_infer
    for (uint32_t i = 0; i < CoarseDepthBinFiles.size(); i++) {
        LogInfo << "Fine_Processing: " + std::to_string(i + 1) + "/" +
            std::to_string(CoarseDepthBinFiles.size()) + " ---> " + CoarseDepthBinFiles[i];
        ret = depthnet->FineProcess(RgbBinPath, RgbBinFiles[i], CoarseDepthBinFiles[i]);
        if (ret != APP_ERR_OK) {
            LogError << "DepthNet fine_process failed, ret=" << ret << ".";
            depthnet->DeInit();
            return ret;
        }
        if (cnt++ % 1000 == 0) {
            LogError << cnt << std::endl;
        }
    }

    // get infer time
    auto endTime = std::chrono::high_resolution_clock::now();
    depthnet->DeInit();
    double costMilliSecs =
        std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * RgbBinFiles.size() / depthnet->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps
        << " imgs/sec";
    return APP_ERR_OK;
}

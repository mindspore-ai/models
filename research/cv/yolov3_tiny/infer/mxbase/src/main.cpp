/*
 * Copyright (c) 2022 Huawei Technologies Co., Ltd.
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
#include <unistd.h>
#include<fstream>
#include<string>
#include "Yolov3_tiny.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t DEVICE_ID = 0;
    const char RESULT_PATH[] = "../data/preds/mxbase";
    const char MODEL_PATH[] = "../convert/yolov3_tiny.om";
}  // namespace

void ShowUsage() {
    LogInfo << "Usage   : ./yolov3_tiny <--image or --dir> [Option]" << std::endl;
    LogInfo << "Options :" << std::endl;
    LogInfo << " --image infer_image_path    the path of single infer image, such as "
               "./yolov3_tiny --image ./test.jpg." << std::endl;
    LogInfo << " --dir infer_image_dir       the dir of batch infer images, such as "
               "./yolov3_tiny --dir ./images." << std::endl;
    return;
}

APP_ERROR ReadImagesPath(const std::string& dir, std::vector<std::string> *imagesPath) {
    DIR *dirPtr = opendir(dir.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir: " << dir;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }
        (*imagesPath).emplace_back(dir + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LogInfo << "Please use as follows." << std::endl;
        ShowUsage();
        return APP_ERR_OK;
    }

    std::string option = argv[1];
    std::string imgPath = argv[2];


    InitParam initParam = {};
    initParam.deviceId = DEVICE_ID;
    initParam.modelPath = MODEL_PATH;
    auto yolov3_tiny = std::make_shared<Yolov3_tiny>();
    APP_ERROR ret = yolov3_tiny->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Yolov3_tiny init failed, ret=" << ret << ".";
        return ret;
    }


    std::vector<std::string> imagesPath;
    if (option == "--image") {
        imagesPath.emplace_back(imgPath);
    } else {
        ret = ReadImagesPath(imgPath, &imagesPath);
    }

    if (ret != APP_ERR_OK) {
        LogInfo << "read file failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "read file success.";

    auto startTime = std::chrono::high_resolution_clock::now();
    for (auto &imgFile : imagesPath) {
        ret = yolov3_tiny->Process(imgFile, RESULT_PATH);
        if (ret != APP_ERR_OK) {
            LogError << "Yolov3_tiny process failed, ret=" << ret << ".";
            yolov3_tiny->DeInit();
            return ret;
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    yolov3_tiny->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imagesPath.size() / yolov3_tiny->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec     "
            << imagesPath.size() << "in Total";
    return APP_ERR_OK;
}

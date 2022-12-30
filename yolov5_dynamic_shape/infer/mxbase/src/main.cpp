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
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "Yolov5Detection.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void ShowUsage() {
    LogInfo << "Usage   : ./yolov5 <--image or --dir> [Option]" << std::endl;
    LogInfo << "Options :" << std::endl;
    LogInfo << " --image infer_image_path    the path of single infer image, such as "
               "./yolov5 --image /home/infer/images/test.jpg." << std::endl;
    LogInfo << " --dir infer_image_dir       the dir of batch infer images, such as "
               "./yolov5 --dir /home/infer/images." << std::endl;
    return;
}

void InitYolov4TinyParam(InitParam *initParam) {
    initParam->deviceId = 0;
    initParam->labelPath = "../../data/models/coco2017.names";
    initParam->checkTensor = true;
    initParam->modelPath = "../../data/models/yolov5.om";
    initParam->classNum = 80;
    initParam->biasesNum = 18;
    initParam->biases = "12,16,19,36,40,28,36,75,76,55,72,146,142,110,192,243,459,401";
    initParam->objectnessThresh = "0.001";
    initParam->iouThresh = "0.6";
    initParam->scoreThresh = "0.001";
    initParam->yoloType = 3;
    initParam->modelType = 0;
    initParam->inputType = 0;
    initParam->anchorDim = 3;
}

APP_ERROR saveResult(const std::vector<std::string> &jsonText, const std::string &savePath) {
    // create result directory when it does not exit
    std::string resultPath = savePath;
    if (access(resultPath.c_str(), 0) != 0) {
        int ret = mkdir(resultPath.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPath << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }
    // create result file under result directory
    resultPath = resultPath + "/predict.json";
    std::ofstream tfile(resultPath, std::ofstream::out|std::ofstream::trunc);
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultPath;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    tfile << "[";
    for (uint32_t i = 0; i < jsonText.size(); i++) {
        tfile << jsonText[i];
        if (i != jsonText.size() - 1) tfile << ", ";
    }
    tfile << "]";
    tfile.close();

    return APP_ERR_OK;
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

    if (option != "--image" && option != "--dir") {
        LogInfo << "Please use as follows." << std::endl;
        ShowUsage();
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    InitYolov4TinyParam(&initParam);
    auto yolov5 = std::make_shared<Yolov5Detection>();
    APP_ERROR ret = yolov5->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogInfo << "Yolov5DetectionOpencv init failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "End to Init yolov5.";

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
    std::vector<std::string> jsonText;
    for (auto path : imagesPath) {
        LogInfo << "read image path " << path;
        yolov5->Process(path, &jsonText);
    }

    std::string resultPathName = "../result/";
    saveResult(jsonText, resultPathName);

    yolov5->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " images/sec.";
    return APP_ERR_OK;
}

/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "AutoDeepLab.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NUM = 19;
    const uint32_t MODEL_TYPE = 1;
    const uint32_t FRAMEWORK_TYPE = 2;
}

void ShowUsage() {
    LogWarn << "Usage   : ./build/autodeeplab <--image or --dir> [Option]" << std::endl;
    LogWarn << "Options :" << std::endl;
    LogWarn << " --image infer_image_path    the path of single infer image, such as "
               "./build/autodeeplab --image /PATH/TO/cityscapes/test.jpg." << std::endl;
    LogWarn << " --dir infer_image_dir       the dir of batch infer images, such as "
               "./build/autodeeplab --dir /PATH/TO/cityscapes/leftImg8bit/val." << std::endl;
    return;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        LogWarn << "Please use as follows." << std::endl;
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
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.modelType = MODEL_TYPE;
    initParam.labelPath = "../model/autodeeplab.names";
    initParam.modelPath = "../model/Auto-DeepLab-s_NHWC_BGR.om";
    initParam.checkModel = true;
    initParam.frameworkType = FRAMEWORK_TYPE;

    AutoDeepLab model;
    APP_ERROR ret = model.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "AutoDeepLab init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::string> imagesPath;
    if (option == "--image") {
        imagesPath.emplace_back(imgPath);
    } else {
        ret = GetAllImages(imgPath, &imagesPath);
    }

    if (ret != APP_ERR_OK) {
        LogError << "read file failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "read file success.";

    for (auto path : imagesPath) {
        LogInfo << "read image path " << path;
        ret = model.Process(path);
        if (ret != APP_ERR_OK) {
            LogError << "AutoDeepLab process failed, ret=" << ret << ".";
            model.DeInit();
            return ret;
        }
    }

    model.DeInit();
    return APP_ERR_OK;
}

/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <fstream>
#include "MxBase/Log/Log.h"
#include "Nasfpn.h"

APP_ERROR ReadImagesPath(const std::string &path, std::vector<std::string> *imagesPath) {
    std::ifstream inFile;
    inFile.open(path, std::ios_base::in);
    std::string line;
    // Check images path file validity
    if (inFile.fail()) {
        LogError << "Failed to open annotation file: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // construct label map
    while (std::getline(inFile, line)) {
        if (line.size() < 10) {
            continue;
        }
        std::string str_path = line;
        imagesPath->push_back(str_path);
    }

    inFile.close();
    return APP_ERR_OK;
}



int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/nasfpn.om";
    std::string dataPath = "../../coco/val2017/";
    std::string annoPath = "../data/data/infer_anno.txt";

    auto model_nasfpn = std::make_shared<nasfpn>();
    APP_ERROR ret = model_nasfpn->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Tagging init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::string> imagesName;
    ret = ReadImagesPath(annoPath, &imagesName);

    if (ret != APP_ERR_OK) {
        model_nasfpn->DeInit();
        return ret;
    }

    int img_size = imagesName.size();
    LogInfo << "test image size:" << img_size;

    for (int i=0; i < img_size; i++) {
        LogInfo << imagesName[i];
        ret = model_nasfpn->Process(imagesName[i], dataPath + imagesName[i] + ".jpg", initParam);
        if (ret !=APP_ERR_OK) {
            LogError << "nasfpn process failed, ret=" << ret << ".";
            model_nasfpn->DeInit();
            return ret;
        }
    }
    model_nasfpn->DeInit();

    return APP_ERR_OK;
}

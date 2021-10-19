/**
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

#include "Stgan.h"

namespace {
    std::vector<std::string> SELECTED_ATTRS {"Bangs", "Blond_Hair", "Mustache", "Young"};
    const std::string OM_MODEL_PATH = "../data/model/stgan_model.om";
}

APP_ERROR ScanImages(const std::string &path, std::vector<std::string> *imgFiles) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
        LogError << "opendir failed. dir:" << path;
        return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
        std::string fileName = direntPtr->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        imgFiles->emplace_back(path + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

std::vector<std::string> split(std::string str, char ch) {
    size_t start = 0;
    size_t len = 0;
    std::vector<std::string> ret;
    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] == ch && i+1 < str.length() && str[i+1] == ch) {
            continue;
        }
        if (str[i] == ch) {
            ret.push_back(str.substr(start, len));
            start = i+1;
            len = 0;
        } else {
            len++;
        }
    }
    if (start < str.length())
        ret.push_back(str.substr(start, len));
    return ret;
}


int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as '../data/test_data/'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = OM_MODEL_PATH;
    initParam.savePath = "../data/mxbase_result";
    auto stgan = std::make_shared<Stgan>();
    APP_ERROR ret = stgan->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Stgan init failed, ret=" << ret << ".";
        return ret;
    }

     // Read the contents of a label
    std::string dataPath = argv[1];
    std::string imagePath = dataPath + "/image/";
    std::string labelPath = dataPath + "/anno/list_attr_celeba.txt";

    std::vector<std::string> imagePathList;
    ret = ScanImages(imagePath, &imagePathList);
    if (ret != APP_ERR_OK) {
        LogError << "Stgan init failed, ret=" << ret << ".";
        return ret;
    }
    std::ifstream fin;
    std::string s;
    fin.open(labelPath);
    int i = 0;
    int imgNum;
    std::map<int, std::string> idx2attr;
    std::map<std::string, int> attr2idx;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (getline(fin, s)) {
        i++;
        if (i == 1) {
            imgNum = atoi(s.c_str());
        } else if (i == 2) {
            std::vector<std::string> allAttrNames = split(s, ' ');
            for (size_t j = 0; j < allAttrNames.size(); j++) {
                idx2attr[j] = allAttrNames[j];
                attr2idx[allAttrNames[j]] = j;
            }
        } else {
            std::vector<std::string> eachAttr = split(s, ' ');
            // first one is file name
            std::string imgName = eachAttr[0];
            std::vector<float> label;
            for (size_t j = 0; j < SELECTED_ATTRS.size(); j++) {
                label.push_back(atoi(eachAttr[attr2idx[SELECTED_ATTRS[j]] + 1].c_str()) * -0.5);
            }
            ret = stgan->Process(imagePath + imgName, imgName, label);
            if (ret != APP_ERR_OK) {
                LogError << "Stgan process failed, ret=" << ret << ".";
                stgan->DeInit();
                return ret;
            }
        }
    }
    fin.close();
    auto endTime = std::chrono::high_resolution_clock::now();
    stgan->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgNum / stgan->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}

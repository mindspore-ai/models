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
#include "Vnet.h"

void SplitString(const std::string &s, std::vector<std::string> *v, const std::string &c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v->push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }

    if (pos1 != s.length()) {
        v->push_back(s.substr(pos1));
    }
}

APP_ERROR ReadImagesPath(const std::string &path, std::vector<std::string> *imagesPath) {
    std::ifstream inFile;
    inFile.open(path, std::ios_base::in);
    std::string line;
    // Check images path file validity
    if (inFile.fail()) {
        LogError << "Failed to open annotation file: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::vector<std::string> vectorStr_path;
    std::string splitStr_path = ".";
    // construct label map
    while (std::getline(inFile, line)) {
        vectorStr_path.clear();

        SplitString(line, &vectorStr_path, splitStr_path);
        std::string str_path = vectorStr_path[0];
        imagesPath->push_back(str_path);
    }

    inFile.close();
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/vnet.om";
    std::string dataPath = "../data/infer_data/img/";
    std::string annoPath = "../data/infer_data/infer_anno.txt";

    auto model_vnet = std::make_shared<vnet>();
    APP_ERROR ret = model_vnet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Tagging init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::string> imagesPath;
    ret = ReadImagesPath(annoPath, &imagesPath);
    if (ret != APP_ERR_OK) {
        model_vnet->DeInit();
        return ret;
    }

    int img_size = imagesPath.size();
    std::vector<int> outputs;
    for (int i=0; i < img_size; i++) {
        ret = model_vnet->Process(imagesPath[i], dataPath + imagesPath[i] + ".bin", initParam, outputs);
        if (ret !=APP_ERR_OK) {
            LogError << "vnet process failed, ret=" << ret << ".";
            model_vnet->DeInit();
            return ret;
        }
    }

    model_vnet->DeInit();

    double total_time = model_vnet->GetInferCostMilliSec() / 1000;
    LogInfo<< "inferance total cost time: "<< total_time<< ", FPS: "<< img_size/total_time;

    return APP_ERR_OK;
}

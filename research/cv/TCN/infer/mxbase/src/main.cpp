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
#include "TCN.h"

namespace {
    const uint32_t CLASS_NUM = 10;
    const uint32_t BATCH_SIZE = 1;
    const std::string resFileName = "../results/eval_mxbase.log";
}  // namespace

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

APP_ERROR ReadImagesPath(const std::string &path, std::vector<std::string> *imagesPath, std::vector<int> *imageslabel) {
    std::ifstream inFile;
    inFile.open(path, std::ios_base::in);
    std::string line;
    // Check images path file validity
    if (inFile.fail()) {
        LogError << "Failed to open annotation file: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::vector<std::string> vectorStr_path;
    std::string splitStr_path = ",";
    // construct label map
    while (std::getline(inFile, line)) {
        vectorStr_path.clear();

        SplitString(line, &vectorStr_path, splitStr_path);
        std::string str_path = vectorStr_path[0];
        std::string str_label = vectorStr_path[1];
        imagesPath->push_back(str_path);
        int label = str_label[0] - '0';
        imageslabel->push_back(label);
    }

    inFile.close();
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/TCN.om";
    std::string dataPath = "../data/mnist_infer_data/mnist_bs_1_bin/";
    std::string annoPath = "../data/mnist_infer_data/mnist_bs_1_label.txt";

    auto model_TCN = std::make_shared<TCN>();
    APP_ERROR ret = model_TCN->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Tagging init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::string> imagesPath;
    std::vector<int> imageslabel;
    ret = ReadImagesPath(annoPath, &imagesPath, &imageslabel);
    if (ret != APP_ERR_OK) {
        model_TCN->DeInit();
        return ret;
    }

    int img_size = imagesPath.size();
    std::vector<int> outputs;
    for (int i=0; i < img_size; i++) {
        ret = model_TCN->Process(dataPath + imagesPath[i], initParam, outputs);
        if (ret !=APP_ERR_OK) {
            LogError << "TCN process failed, ret=" << ret << ".";
            model_TCN->DeInit();
            return ret;
        }
    }
    float cor = 0;
    for (int i = 0; i < img_size; i++) {
        int label_now = imageslabel[i];
        if (label_now == outputs[i]) {
            cor++;
        }
    }

    model_TCN->DeInit();

    double total_time = model_TCN->GetInferCostMilliSec() / 1000;
    LogInfo<< "total num: "<< img_size<< ",acc total: "<< static_cast<float>(cor/img_size);
    LogInfo<< "inferance total cost time: "<< total_time<< ", FPS: "<< img_size/total_time;

    std::ofstream outfile(resFileName);
    if (outfile.fail()) {
    LogError << "Failed to open result file: ";
    return APP_ERR_COMM_FAILURE;
    }
    outfile << "total num: "<< img_size<< ",acc total: "<< static_cast<float>(cor/img_size)<< "\n";
    outfile << "inferance total cost time(s): "<< total_time<< ", FPS: "<< img_size/total_time;
    outfile.close();

    return APP_ERR_OK;
}

/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "Cnndirection.h"

namespace {
    const uint32_t CLASS_NUM = 2;
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
    std::vector<std::string> vectorStr_label;
    std::string splitStr_path = "\t";
    std::string splitStr_label = "_";
    // construct label map
    while (std::getline(inFile, line)) {
        if (line.size() < 10) {
            continue;
        }
        vectorStr_path.clear();
        SplitString(line, &vectorStr_path, splitStr_path);
        std::string str_path = vectorStr_path[0];
        str_path = str_path.replace(str_path.find("\\"), 1, "/").replace(str_path.find("\\"), 1, "/");
        imagesPath->push_back(str_path);
        vectorStr_label.clear();
        SplitString(vectorStr_path[0], &vectorStr_label, splitStr_label);
        int label = vectorStr_label[1][0] - '0';
        imageslabel->push_back(label);
    }

    inFile.close();
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/models/cnn.om";
    std::string dataPath = "../data/image/";
    std::string annoPath = "../data/image/annotation_test.txt";

    auto model_cnndirection = std::make_shared<cnndirection>();
    APP_ERROR ret = model_cnndirection->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Tagging init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::string> imagesPath;
    std::vector<int> imageslabel;
    ret = ReadImagesPath(annoPath, &imagesPath, &imageslabel);

    if (ret != APP_ERR_OK) {
        model_cnndirection->DeInit();
        return ret;
    }

    int img_size = imagesPath.size();
    LogInfo << "test image size:" << img_size;

    std::vector<int> outputs;
    for (int i=0; i < img_size; i++) {
        ret = model_cnndirection->Process(dataPath + imagesPath[i], initParam, outputs);
        if (ret !=APP_ERR_OK) {
            LogError << "Tacotron2 process failed, ret=" << ret << ".";
            model_cnndirection->DeInit();
            return ret;
        }
    }
    float num_0 = 0;
    float num_1 = 0;
    float cor_0 = 0;
    float cor_1 = 0;
    for (int i = 0; i < img_size; i++) {
        int label_now = imageslabel[i];
        if (label_now == 0) {
            num_0++;
            if (outputs[i] == 0) {
                cor_0++;
            }
        } else {
            num_1++;
            if (outputs[i] == 1) {
                cor_1++;
            }
        }
    }

    model_cnndirection->DeInit();

    double total_time = model_cnndirection->GetInferCostMilliSec() / 1000;

    LogInfo<< "num1: "<< num_1<< ",acc1: "<< static_cast<float>(cor_1/num_1);
    LogInfo<< "num0: "<< num_0<< ",acc0: "<< static_cast<float>(cor_0/num_0);
    LogInfo<< "total num: "<< img_size<< ",acc total: "<< static_cast<float>(cor_1+cor_0)/img_size;
    LogInfo<< "inferance total cost time: "<< total_time<< ", FPS: "<< img_size/total_time;

    std::ofstream outfile(resFileName);
    if (outfile.fail()) {
    LogError << "Failed to open result file: ";
    return APP_ERR_COMM_FAILURE;
    }
    outfile << "num1: "<< num_1<< ",acc1: "<< static_cast<float>(cor_1/num_1)<< "\n";
    outfile << "num0: "<< num_0<< ",acc0: "<< static_cast<float>(cor_0/num_0)<< "\n";
    outfile << "total num: "<< img_size<< ",acc total: "<< static_cast<float>(cor_1+cor_0)/img_size<< "\n";
    outfile << "inferance total cost time(s): "<< total_time<< ", FPS: "<< img_size/total_time;
    outfile.close();

    return APP_ERR_OK;
}

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

#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "Gpt2.h"
#include "MxBase/Log/Log.h"

using std::fstream;
using std::ios;

void SplitString(const std::string &s, std::vector<uint32_t> *v, const std::string &c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v->push_back(std::stoi(s.substr(pos1, pos2 - pos1)));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }

    if (pos1 != s.length()) {
        v->push_back(std::stoi(s.substr(pos1)));
    }
}

APP_ERROR ReadtxtPath(const std::string &path, std::vector<std::vector<uint32_t>> *data) {
    std::ifstream inFile;
    inFile.open(path, std::ios_base::in);
    std::string line;
    // Check images path file validity
    if (inFile.fail()) {
        LogError << "Failed to open annotation file: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::vector<std::uint32_t> data_line;
    std::string splitStr_path = " ";
    // construct label map
    while (std::getline(inFile, line)) {
        data_line.clear();
        SplitString(line, &data_line, splitStr_path);
        data->push_back(data_line);
    }

    inFile.close();
    return APP_ERR_OK;
}

APP_ERROR ReadResult(const std::string &path, double &avg_PPL) {
    std::ifstream inFile;
    std::string resultPath = path + "results/result.txt";
    inFile.open(resultPath, std::ios_base::in);
    if (inFile.fail()) {
        LogError << "Failed to open annotation file: " << resultPath;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::uint32_t num = 0;
    double PPL_example = 0;
    while (1) {
        inFile >> PPL_example;
        if (inFile.eof() != 0) {break;}
        avg_PPL += PPL_example;
        num++;
    }
    avg_PPL /= num;

    LogInfo << "================    PPL Calculate    ===============";
    LogInfo << " | Average Loss : " << avg_PPL << ".";
    LogInfo << " | Average PPL : " <<exp(avg_PPL) << ".";
    LogInfo << "====================================================";

    return APP_ERR_OK;
}

APP_ERROR fileEmpty(const std::string &inferPath) {
    std::string resultfileName = inferPath + "results/result.txt";
    fstream file(resultfileName, ios::out);
    std::string scorefileName = inferPath + "results/score.txt";
    fstream file_score(scorefileName, ios::out);

    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    std::string dataPath = "../data/data/";
    std::string inferPath = "../mxbase/";

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../data/model/gpt2.om";

    auto model_gpt2 = std::make_shared<gpt2>();
    APP_ERROR ret = model_gpt2->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "gpt2 init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::vector<uint32_t>> input_ids;
    std::vector<std::vector<uint32_t>> input_mask;
    std::vector<std::vector<uint32_t>> label_ids;

    ret = ReadtxtPath(dataPath + "/input_ids.txt", &input_ids);
    if (ret != APP_ERR_OK) {
        model_gpt2->DeInit();
        return ret;
    }
    ret = ReadtxtPath(dataPath + "/input_mask.txt", &input_mask);
    if (ret != APP_ERR_OK) {
        model_gpt2->DeInit();
        return ret;
    }
    ret = ReadtxtPath(dataPath + "/label_ids.txt", &label_ids);
    if (ret != APP_ERR_OK) {
        model_gpt2->DeInit();
        return ret;
    }

    initParam.seq_len = input_ids[0].size();
    uint32_t txt_size = input_ids.size();
    LogInfo << "test image size:" << txt_size;

    double outputs = 0;

    ret = fileEmpty(inferPath);
    if (ret != APP_ERR_OK) {
    LogError << "fileEmpty is failed, ret=" << ret << ".";
    return ret;
    }

    for (uint32_t i=0; i < txt_size; i++) {
        LogInfo << i;
        ret = model_gpt2->Process(inferPath, input_ids[i], input_mask[i], label_ids[i],
                                  initParam, outputs);
        if (ret !=APP_ERR_OK) {
            LogError << "gpt2 process failed, ret=" << ret << ".";
            model_gpt2->DeInit();
            return ret;
        }
    }

    model_gpt2->DeInit();
    double total_time = model_gpt2->GetInferCostMilliSec() / 500;

    double avg_loss = 0;
    ret = ReadResult(inferPath, avg_loss);
    if (ret != APP_ERR_OK) {
    LogError << "gpt2 cal PPL failed, ret=" << ret << ".";
    return ret;
    }

    LogInfo<< "inferance total cost time: "<< total_time<< ", FPS: "<< txt_size / total_time;

    return APP_ERR_OK;
}



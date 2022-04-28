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
#include "Rotate.h"


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

int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../../data/out/rotate-head.om";
    std::string dataPath = "../../data/wn18rr";

    auto model_rotate_head = std::make_shared<rotate>();
    APP_ERROR ret = model_rotate_head->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Tagging init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::vector<uint32_t>> positive_sample_head;
    std::vector<std::vector<uint32_t>> negative_sample_head;
    std::vector<std::vector<uint32_t>> filter_bias_head;
    ReadtxtPath(dataPath + "/positive_sample_head.txt", &positive_sample_head);
    ReadtxtPath(dataPath + "/negative_sample_head.txt", &negative_sample_head);
    ReadtxtPath(dataPath + "/filter_bias_head.txt", &filter_bias_head);

    initParam.seq_len = negative_sample_head[0].size();
    uint32_t txt_size = positive_sample_head.size();
    LogInfo << "test image size:" << txt_size;

    std::vector<uint32_t> outputs1;
    uint32_t outputs2 = 0;

    for (int i=0; i < txt_size; i++) {
        LogInfo << i;
        APP_ERROR ret1 = model_rotate_head->Process(positive_sample_head[i], negative_sample_head[i],
                                                    filter_bias_head[i], initParam, outputs1, outputs2);
        if (ret1 !=APP_ERR_OK) {
            LogError << "Rotate process failed, ret=" << ret1 << ".";
            model_rotate_head->DeInit();
            return ret1;
        }
    }

    initParam.modelPath = "../../data/out/rotate-tail.om";

    auto model_rotate_tail = std::make_shared<rotate>();
    APP_ERROR ret2 = model_rotate_tail->Init(initParam);
    if (ret2 != APP_ERR_OK) {
        LogError << "Tagging init failed, ret=" << ret2 << ".";
        return ret2;
    }

    std::vector<std::vector<uint32_t>> positive_sample_tail;
    std::vector<std::vector<uint32_t>> negative_sample_tail;
    std::vector<std::vector<uint32_t>> filter_bias_tail;
    ReadtxtPath(dataPath + "/positive_sample_tail.txt", &positive_sample_tail);
    ReadtxtPath(dataPath + "/negative_sample_tail.txt", &negative_sample_tail);
    ReadtxtPath(dataPath + "/filter_bias_tail.txt", &filter_bias_tail);

    for (int i=0; i < txt_size; i++) {
        LogInfo << i;
        APP_ERROR ret3 = model_rotate_tail->Process(positive_sample_tail[i], negative_sample_tail[i],
                                                    filter_bias_tail[i], initParam, outputs1, outputs2);
        if (ret3 !=APP_ERR_OK) {
            LogError << "Rotate process failed, ret=" << ret3 << ".";
            model_rotate_tail->DeInit();
            return ret3;
        }
    }

    model_rotate_head->DeInit();
    double total_time = (model_rotate_head->GetInferCostMilliSec() + model_rotate_tail->GetInferCostMilliSec()) / 1000;
    LogInfo<< "inferance total cost time: "<< total_time<< ", FPS: "<< 2 * txt_size / total_time;

    return APP_ERR_OK;
}


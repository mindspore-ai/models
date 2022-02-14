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
#include "duconv.h" // NOLINT

namespace {
    const uint32_t CLASS_NUM = 2;
    const uint32_t BATCH_SIZE = 1;
    const std::string resFileName = "./results/eval_mxbase.log";
}  // namespace

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
    initParam.modelPath = "../data/model/duconv.om";
    std::string dataPath = "../data/data";

    auto model_duconv = std::make_shared<duconv>();
    APP_ERROR ret = model_duconv->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Tagging init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::vector<uint32_t>> context_id;
    std::vector<std::vector<uint32_t>> context_pos_id;
    std::vector<std::vector<uint32_t>> context_segment_id;
    std::vector<std::vector<uint32_t>> kn_id;
    std::vector<std::vector<uint32_t>> kn_segment_id;
    ret = ReadtxtPath(dataPath + "/context_id.txt", &context_id);
    if (ret != APP_ERR_OK) {
        model_duconv->DeInit();
        return ret;
    }
    ret = ReadtxtPath(dataPath + "/context_pos_id.txt", &context_pos_id);
    if (ret != APP_ERR_OK) {
        model_duconv->DeInit();
        return ret;
    }
    ret = ReadtxtPath(dataPath + "/context_segment_id.txt", &context_segment_id);
    if (ret != APP_ERR_OK) {
        model_duconv->DeInit();
        return ret;
    }
    ret = ReadtxtPath(dataPath + "/kn_id.txt", &kn_id);
    if (ret != APP_ERR_OK) {
        model_duconv->DeInit();
        return ret;
    }
    ret = ReadtxtPath(dataPath + "/kn_seq_length.txt", &kn_segment_id);

    if (ret != APP_ERR_OK) {
        model_duconv->DeInit();
        return ret;
    }

    initParam.seq_len = context_id[0].size();
    uint32_t txt_size = context_id.size();
    LogInfo << "test image size:" << txt_size;

    float outputs = 0;

    for (int i=0; i < txt_size; i++) {
        LogInfo << i;
        ret = model_duconv->Process(context_id[i], context_pos_id[i], context_segment_id[i],
                                    kn_id[i], kn_segment_id[i], initParam, outputs);
        if (ret !=APP_ERR_OK) {
            LogError << "duconv process failed, ret=" << ret << ".";
            model_duconv->DeInit();
            return ret;
        }
    }

    model_duconv->DeInit();
    double total_time = model_duconv->GetInferCostMilliSec() / 500;
    LogInfo<< "inferance total cost time: "<< total_time<< ", FPS: "<< txt_size / total_time;

    return APP_ERR_OK;
}// NOLINT
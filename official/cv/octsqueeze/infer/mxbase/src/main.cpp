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
#include "Octsqueeze.h"


namespace {
    uint32_t MAX_LENGTH = 1000;
}  // namespace

void SplitString(const std::string &s, std::vector<float> *v, const std::string &c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    std::string pos;
    while (std::string::npos != pos2) {
        pos = s.substr(pos1, pos2 - pos1);
        v->push_back(std::stof(pos));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }

    if (pos1 != s.length()) {
        pos = s.substr(pos1);
        v->push_back(std::stof(pos));
    }
}

APP_ERROR ReadtxtPath(const std::string &path, std::vector<std::vector<float>> *data) {
    std::ifstream inFile;
    inFile.open(path, std::ios_base::in);
    std::string line;
    // Check images path file validity
    if (inFile.fail()) {
        LogError << "Failed to open file: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::vector<float> data_line;
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
    initParam.modelPath = "../data/model/octsqueeze.om";
    std::string dataPath = "../data/infernece_dataset/";

    std::vector<std::string> files;
    for (int i = 7000; i < 7100; i++) {
        files.push_back(std::to_string(i));
    }
    std::vector<std::vector<float>> inputs = {};
    float *outputs;
    uint32_t txt_size = 100;
    std::vector<std::string> precision = {"0.01", "0.02", "0.04", "0.08"};
    std::string prc;

    auto model_octsqueeze = std::make_shared<octsqueeze>();
    APP_ERROR ret = model_octsqueeze->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Tagging init failed, ret=" << ret << ".";
        return ret;
    }
    for (int m = 0; m < precision.size(); m++) {
        txt_size = files.size();
        LogInfo << "test data size:" << txt_size;
        prc = precision[m];
        uint32_t idx = 0;
        uint32_t number = 0;

        for (int i = 0; i < txt_size; i++) {
            ReadtxtPath(dataPath + "/" + prc + "/00" + files[i] + "/" + "input.txt", &inputs);
            LogInfo << files[i];
            uint32_t num_points = inputs.size();
            std::string argsort_path = "./results/" + prc + "/00" + files[i] + "_results.bin";
            std::ofstream outfile(argsort_path, std::ios::binary);
            if (outfile.fail()) {
                LogError << "Failed to open result file: ";
                return APP_ERR_COMM_FAILURE;
            }
            std::vector<std::vector<float>> input = {};
            while (number < num_points + 1) {
                if (idx < MAX_LENGTH && number < num_points) {
                    input.push_back(inputs[number]);
                    idx++;
                    number++;

                } else {
                    APP_ERROR ret1 =  model_octsqueeze->Process(input, initParam, outputs);
                    if (ret1 !=APP_ERR_OK) {
                        LogError << "octsqueeze process failed, ret=" << ret1 << ".";
                        model_octsqueeze->DeInit();
                        return ret1;
                    }
                    uint32_t index = 0;
                    outfile.write(reinterpret_cast<char*>(outputs), sizeof(float) * idx * 256);
                    idx = 0;
                    input.clear();
                    if (number == num_points) {
                        number++;
                    }
                }
            }
            outfile.close();
        }
    }

    model_octsqueeze->DeInit();
    double total_time = model_octsqueeze->GetInferCostMilliSec() / 1000;
    LogInfo<< "inferance total cost time: "<< total_time<< ", FPS: "<< txt_size / total_time;

    return APP_ERR_OK;
}

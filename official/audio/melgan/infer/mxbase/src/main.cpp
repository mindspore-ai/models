/*
 * Copyright 2022 Huawei Technologies Co., Ltd.
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
#include <string>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstdio>
#include "Melgan.h"
#include "MxBase/Log/Log.h"

int eval_length = 240;
int hop_size = 256;
int repeat_frame = 30;
int sample = 22050;


APP_ERROR ReadTxt(const std::string &path, std::vector<std::vector<std::vector<float>>> *dataset) {
    std::ifstream fp(path);
    std::string line;
    std::vector<std::vector<float>> data;
    int count = 0;
    while (std::getline(fp, line)) {
        std::vector<float> data_line;
        std::string number;
        std::istringstream readstr(line);
        for (int j = 0; j < 240; j++) {
            std::getline(readstr, number, ' ');
            data_line.push_back(static_cast<float>(atof(number.c_str())));
        }
        data.push_back(data_line);
        count++;
        if (count % 80 == 0) {
            std::vector<std::vector<float>> dataseg;
            for (int i = count - 80; i < count; i++) {
                dataseg.push_back(data[i]);
            }
            dataset->push_back(dataseg);
        }
    }
    return APP_ERR_OK;
}


int main(int argc, char *argv[]) {
    std::string model_path = argv[1];
    std::string eval_data_path = argv[2];
    std::string list_filename = argv[3];

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = model_path;

    auto melgan = std::make_shared<MELGAN>();
    printf("Start running\n");
    APP_ERROR ret = melgan->Init(initParam);
    if (ret != APP_ERR_OK) {
        melgan->DeInit();
        LogError << "melgan init failed, ret=" << ret << ".";
        return ret;
    }

    // get test data filename
    std::string path = eval_data_path + "/" + list_filename;
    std::ifstream fp(path);
    std::string filename;
    while (std::getline(fp, filename)) {
        LogInfo << "Start inference " << filename << std::endl;
        std::string dataPath = eval_data_path + "/" + filename;
        std::vector<std::vector<std::vector<float>>> test_data;

        ret = ReadTxt(dataPath, &test_data);
        if (ret != APP_ERR_OK) {
            melgan->DeInit();
            LogError << "read test_data failed, ret=" << ret << ".";
            return ret;
        }

        int data_seg = test_data.size();
        int data_row = test_data[0].size();
        int data_col = test_data[0][0].size();
        LogInfo << filename << "data shape: (" << data_seg << ',' << data_row << ',' << data_col << ')';
        for (int iter = 0; iter < data_seg; iter++) {
            std::vector<float> output;
            std::vector<std::vector<std::vector<float>>> data;
            data.push_back(test_data[iter]);
            ret = melgan->Process(filename, data, initParam, output);
            if (ret != APP_ERR_OK) {
                LogError << "melgan process failed, ret=" << ret << ".";
                melgan->DeInit();
                return ret;
            }
        }
        LogInfo << "File " << filename << " inference successfully!";
    }

    melgan->DeInit();
    return APP_ERR_OK;
}

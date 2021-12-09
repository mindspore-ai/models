/*
 * Copyright 2021 Huawei Technologies Co., Ltd.
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
#include "Dscnn.h"
#include "MxBase/Log/Log.h"

APP_ERROR ReadTxt(const std::string &path, std::vector<std::vector<float>> *dataset) {
    std::ifstream fp(path);
    std::string line;
    while (std::getline(fp, line)) {
        std::vector<float> data_line;
        std::string number;
        std::istringstream readstr(line);
        for (int j = 0; j < 980; j++) {
            std::getline(readstr, number, ' ');
            data_line.push_back(atof(number.c_str()));
        }
        dataset->push_back(data_line);
    }
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/models/dscnn.om";
    auto dscnn = std::make_shared<DSCNN>();
    printf("Start running\n");
    APP_ERROR ret = dscnn->Init(initParam);
    if (ret != APP_ERR_OK) {
        dscnn->DeInit();
        LogError << "dscnn init failed, ret=" << ret << ".";
        return ret;
    }

    std::string dataPath = "../data/input/validation_data.txt";
    std::string labelPath = "../data/input/validation_label.txt";

    std::vector<std::vector<float>> test_data;
    std::vector<std::vector<float>> test_label;

    ret = ReadTxt(dataPath, &test_data);
    if (ret != APP_ERR_OK) {
        dscnn->DeInit();
        LogError << "read test_data failed, ret=" << ret << ".";
        return ret;
    }

    ret = ReadTxt(labelPath, &test_label);
    if (ret != APP_ERR_OK) {
        dscnn->DeInit();
        LogError << "read test_label failed, ret=" << ret << ".";
        return ret;
    }

    int data_row = test_data.size();
    int data_col = data_row > 0 ? test_data[0].size():0;

    if (data_col == 0) {
        LogError << "dscnn process testing   data failed, data_col=" << data_col << ".";
        dscnn->DeInit();
        return -1;
    }
    std::vector<int> output1;
    std::vector<std::vector<int>> output5;
    for (int i=0; i < data_row; i++) {
        std::vector<std::vector<float>> data;
        std::vector<std::vector<float>> label;
        data.emplace_back(test_data[i]);
        label.emplace_back(test_label[i]);
        ret = dscnn->Process(data, initParam, &output1, &output5);
        if (ret !=APP_ERR_OK) {
            LogError << "dscnn process failed, ret=" << ret << ".";
            dscnn->DeInit();
            return ret;
        }
    }
    int corr1 = 0;
    int corr5 = 0;
    for (int i = 0; i < data_row; i++) {
        if (test_label[i][0] == output1[i]) corr1++;
        for (int j = 0; j < 5 ; j++) {
            if (test_label[i][0] == output5[i][j]) {
                corr5++;
                break;
            }
        }
    }
    float acc1 = 100.0 * corr1 / data_row;
    float acc5 = 100.0 * corr5 / data_row;
    printf("Eval: top1_cor:%d, top5_cor:%d, tot:%d, acc@1=%.2f%%, acc@5=%.2f%%\n",
           corr1, corr5, data_row, acc1, acc5);
    dscnn->DeInit();
    return APP_ERR_OK;
}


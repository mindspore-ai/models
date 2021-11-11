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
#include <string>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>
#include "stgcnUtil.h"
#include "MxBase/Log/Log.h"

APP_ERROR ReadCsv(const std::string &path, std::vector<std::vector<float>> &dataset) {
    std::ifstream fp(path);
    std::string line;
    while (std::getline(fp, line)) {
        std::vector<float> data_line;
        std::string number;
        std::istringstream readstr(line);

        for (int j = 0; j < 228; j++) {
            std::getline(readstr, number, ',');
            data_line.push_back(atof(number.c_str()));
        }
        dataset.push_back(data_line);
    }
    return APP_ERR_OK;
}

APP_ERROR transform(std::vector<std::vector<float>>& dataset,
    const std::vector<float>& mean, const std::vector<float>& stdd) {
    for (uint32_t i = 0; i < dataset.size(); ++i) {
        for (uint32_t j = 0; j < dataset[0].size(); ++j) {
            dataset[i][j] = (dataset[i][j]-mean[j])/sqrt(stdd[j]);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR getMeanStd(std::vector<std::vector<float>> dataset, std::vector<float>& mean, std::vector<float>& stdd) {
    for (uint32_t j = 0; j < dataset[0].size(); ++j) {
        float m = 0.0;
        float var = 0.0;
        for (uint32_t i = 0; i < dataset.size(); ++i) {
            m += dataset[i][j];
        }
        m /= dataset.size();
        for (uint32_t i = 0; i < dataset.size(); ++i) {
            var += (dataset[i][j] - m)*(dataset[i][j] - m);
        }
        var /= (dataset.size());

        mean.emplace_back(m);
        stdd.emplace_back(var);
    }
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input dataset path and n_pred, such as './data/vel/csv 9'";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/models/stgcn.om";
    auto stgcn = std::make_shared<STGCN>();
    APP_ERROR ret = stgcn->Init(initParam);
    if (ret != APP_ERR_OK) {
        stgcn->DeInit();
        LogError << "stgcn init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    int n_pred = atoi(argv[2]);

    std::vector<std::vector<float>> dataset;

    ret = ReadCsv(imgPath, dataset);
    if (ret != APP_ERR_OK) {
        stgcn->DeInit();
        LogError << "read dataset failed, ret=" << ret << ".";
        return ret;
    }

    float val_and_test_rate = 0.15;
    int data_row = dataset.size();
    int data_col = data_row > 0 ? dataset[0].size():0;

    if (data_col == 0) {
        LogError << "stgcn process dataset failed, data_col=" << data_col << ".";
        stgcn->DeInit();
        return -1;
    }

    int len_val = static_cast<int>(floor(data_row * val_and_test_rate));
    int len_test = static_cast<int>(floor(data_row * val_and_test_rate));
    int len_train = static_cast<int>(data_row - len_val - len_test);

    std::vector<std::vector<float>> dataset_train;
    std::vector<std::vector<float>> dataset_test;

    for (int i = 0; i < data_row; ++i) {
        if (i < len_train) {
            dataset_train.emplace_back(dataset[i]);
        } else if (i >= (len_train + len_val)) {
            dataset_test.emplace_back(dataset[i]);
        } else {
            continue;
        }
    }

    ret = getMeanStd(dataset_train, initParam.MEAN, initParam.STD);
    if (ret != APP_ERR_OK) {
        LogError << "get mean and std of train dataset failed, ret=" << ret << ".";
        return ret;
    }

    // Norlize test dataset
    ret = transform(dataset_test, initParam.MEAN, initParam.STD);
    if (ret != APP_ERR_OK) {
        LogError << "transform test dataset failed, ret=" << ret << ".";
        return ret;
    }

    int n_his = 12;
    int num = dataset_test.size() - n_his - n_pred;

    for (int i=0; i < num; ++i) {
        std::vector<std::vector<float>> data;
        for (int t = i; t < i + n_his; ++t) {
            data.emplace_back(dataset_test[t]);
        }

        ret = stgcn->Process(data, initParam);
        if (ret != APP_ERR_OK) {
            LogError << "stgcn process failed, ret=" << ret << ".";
            stgcn->DeInit();
            return ret;
        }
    }

    stgcn->DeInit();
    return APP_ERR_OK;
}


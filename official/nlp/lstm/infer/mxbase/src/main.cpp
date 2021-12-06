/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
#include <vector>
#include "SentimentNet.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_infer_cost;
uint32_t g_true_positive = 0;
uint32_t g_false_positive = 0;
uint32_t g_true_negative = 0;
uint32_t g_false_negative = 0;

static void InitSentimentNetParam(std::shared_ptr<InitParam> initParam) {
    initParam->deviceId = 0;
    initParam->modelPath = "../model/LSTM.om";
}

static APP_ERROR ReadFilesFromPath(const std::string &path, std::shared_ptr<std::vector<std::string>> files) {
    DIR *dir = NULL;
    struct dirent *ptr = NULL;

    if ((dir = opendir(path.c_str())) == NULL) {
        LogError << "Open dir error: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }

    while ((ptr = readdir(dir)) != NULL) {
        // d_type == 8 is files
        if (ptr->d_type == 8) {
            files->push_back(ptr->d_name);
        }
    }
    closedir(dir);
    // sort ascending order
    sort(files->begin(), files->end());
    return APP_ERR_OK;
}

static void CalculateMetrics() {
    // calculate metrics
    LogInfo << "==============================================================";
    uint32_t total = g_true_positive + g_false_positive + g_true_negative + g_false_negative;
    float precision = (g_true_positive + g_true_negative) * 1.0 / total;
    float posPrecision = g_true_positive * 1.0 / (g_true_positive + g_false_positive);
    float negPrecision = g_true_negative * 1.0 / (g_true_negative + g_false_negative);
    LogInfo << "total accuracy: " << precision;
    LogInfo << "positive precision: " << posPrecision;
    LogInfo << "negative precision: " << negPrecision;
    float posRecall = g_true_positive * 1.0 / (g_true_positive + g_false_negative);
    float negRecall = g_true_negative * 1.0 / (g_true_negative + g_false_positive);
    LogInfo << "positive recall: " << posRecall;
    LogInfo << "negative recall: " << negRecall;
    LogInfo << "positive F1 Score: " << 2 * posPrecision * posRecall / (posPrecision + posRecall);
    LogInfo << "negative F1 Score: " << 2 * negPrecision * negRecall / (negPrecision + negRecall);
    LogInfo << "==============================================================";
}

int main(int argc, char *argv[]) {
    bool eval = false;
    uint32_t maxLoadNum = 2;
    std::string inferPath = "";
    std::string labelPath = "";

    int numArgTwo = 2;
    int numArgThree = 3;
    int numArgFour = 4;
    int numArgFive = 5;
    if (argc <= 1) {
        LogWarn << "Please input sentences file path, such as './lstm /input/data 2 true /input/label.txt'.";
        return APP_ERR_OK;
    } else if (argc == numArgTwo) {
        inferPath = argv[1];
    } else if (argc == numArgThree) {
        inferPath = argv[1];
        maxLoadNum = atoi(argv[2]);
    } else if (argc == numArgFour) {
        inferPath = argv[1];
        eval = atoi(argv[2]);
        labelPath = argv[3];
    } else if (argc == numArgFive) {
        inferPath = argv[1];
        maxLoadNum = atoi(argv[2]);
        eval = atoi(argv[3]);
        labelPath = argv[4];
    }

    if (inferPath == "") {
        LogWarn << "Input sentences dir is null, use default config";
        inferPath = "../dataset/aclImdb/preprocess/00_data/";
    }
    if (eval && labelPath == "") {
        LogWarn << "Input sentences label path is null, use default config";
        labelPath = "../dataset/aclImdb/preprocess/labels.txt";
    }

    // load sentences files
    std::shared_ptr<std::vector<std::string>> files = std::make_shared<std::vector<std::string>>();
    APP_ERROR ret = ReadFilesFromPath(inferPath, files);
    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "test set files: " << std::to_string(files->size());

    // load sentences labels
    std::vector<uint32_t> labels;
    if (eval) {
        std::ifstream labelFile(labelPath);
        std::string line;
        while (std::getline(labelFile, line)) {
            labels.push_back(atoi(line.c_str()));
        }
        labelFile.close();

        LogInfo << "test set size: " << std::to_string(labels.size());
    }

    // init SentimentNet
    std::shared_ptr<InitParam> initParam = std::make_shared<InitParam>();
    InitSentimentNetParam(initParam);
    auto sentimentNet = std::make_shared<SentimentNet>();
    ret = sentimentNet->Init(*(initParam.get()));
    if (ret != APP_ERR_OK) {
        LogError << "SentimentNet init failed, ret=" << ret << ".";
        return ret;
    }

    // infer
    const uint32_t batchSize = 64;
    LogInfo << "Max test num: " << std::to_string(maxLoadNum) << " data batch size: " << std::to_string(batchSize);
    bool firstInput = true;
    for (uint32_t i = 0; i < files->size(); i++) {
        if (i + 1 > maxLoadNum) {
            break;
        }
        std::string fileName = "LSTM_data_bs64_" + std::to_string(i) + ".bin";
        LogInfo << "read file name: " << fileName;
        ret = sentimentNet->Process(inferPath, fileName, firstInput, eval, labels, i * batchSize);
        firstInput = false;
        if (ret != APP_ERR_OK) {
            LogError << "SentimentNet process failed, ret=" << ret << ".";
            sentimentNet->DeInit();
            return ret;
        }
    }

    // evaluate and statistic delay
    if (eval) {
        CalculateMetrics();
    }
    double costSum = 0;
    for (uint32_t i = 0; i < g_infer_cost.size(); i++) {
        costSum += g_infer_cost[i];
    }
    LogInfo << "Infer sentences sum " << g_infer_cost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_infer_cost.size() * 1000 / costSum << " bin/sec.";

    // DeInit SentimentNet
    ret = sentimentNet->DeInit();
    if (ret != APP_ERR_OK) {
        LogError << "SentimentNet DeInit failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

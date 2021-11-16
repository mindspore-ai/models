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
#include "TextrcnnBase.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_infer_cost;
uint32_t g_tp = 0;
uint32_t g_fp = 0;
uint32_t g_fn = 0;
uint32_t g_tn = 0;

void init_textrcnn_param(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->labelPath = "../data/config/infer_label.txt";
    initParam->modelPath = "../data/model/textrcnn.om";
    initParam->classNum = 2;
}

APP_ERROR read_files_from_path(const std::string &path, std::vector<std::string> *files) {
    DIR *dir = NULL;
    struct dirent *ptr = NULL;

    if ((dir = opendir(path.c_str())) == NULL) {
        LogError << "Open dir error: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }

    while ((ptr = readdir(dir)) != NULL) {
        // d_type == 8 is file
        if (ptr->d_type == 8) {
            files->push_back(ptr->d_name);
        }
    }
    closedir(dir);
    // sort ascending order
    sort(files->begin(), files->end());
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './textrcnn ../data/MR 1'.";
        return APP_ERR_OK;
    }
    InitParam initParam;
    init_textrcnn_param(&initParam);
    auto textrcnnBase = std::make_shared<TextrcnnBase>();
    APP_ERROR ret = textrcnnBase->init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Textrcnnbase init failed, ret=" << ret << ".";
        return ret;
    }

    // argv[1] is the parameter what the file path is.
    std::string inferPath = argv[1];
    std::vector<std::string> files;
    ret = read_files_from_path(inferPath + "/00_feature/", &files);

    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        return ret;
    }

    // argv[2] is the parameter whether do accuracy evaluation.
    bool eval = atoi(argv[2]);

    LogInfo << files.size();
    // Inference for all binary files
    for (uint32_t i = 0; i < files.size(); i++) {
        // call the function which arranges the whole process of one binary file.
        ret = textrcnnBase->process(inferPath, files[i], eval);
        if (ret != APP_ERR_OK) {
            LogError << "Textrcnnbase process failed, ret=" << ret << ".";
            textrcnnBase->de_init();
            return ret;
        }
    }

    if (eval) {
        LogInfo << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
        float accuracy = (g_tp + g_tn) * 1.0 / (g_tp + g_fp + g_fn +  g_tn);
        LogInfo << "Accuracy: " << accuracy;
        float precision = g_tp * 1.0 / (g_tp + g_fp);
        LogInfo << "Precision: " << precision;
        float recall = g_tp * 1.0 / (g_tp + g_fn);
        LogInfo << "Recall: " << recall;
        LogInfo << "F1 Score: " << 2 * precision * recall / (precision + recall);
        LogInfo << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
    }
    textrcnnBase->de_init();
    double costSum = 0;
    for (uint32_t i = 0; i < g_infer_cost.size(); i++) {
        costSum += g_infer_cost[i];
    }
    LogInfo << "Infer words sum " << g_infer_cost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_infer_cost.size() * 1000 / costSum << " bin/sec.";
    return APP_ERR_OK;
}

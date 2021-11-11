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

#include "TextCnnBase.h"
#include <dirent.h>
#include "MxBase/Log/Log.h"

std::vector<double> g_infer_cost;
uint32_t g_tp = 0;
uint32_t g_fp = 0;
uint32_t g_fn = 0;

void init_textcnn_param(InitParam *initParam) {
    initParam->deviceId = 0;
    initParam->labelPath = "../data/config/infer_label.txt";
    initParam->modelPath = "../data/model/textcnn.om";
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

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './textcnn ../data/ 1'";
        return APP_ERR_OK;
    }

    InitParam initParam;
    init_textcnn_param(&initParam);
    auto textcnnBase = std::make_shared<TextCnnBase>();
    APP_ERROR ret = textcnnBase->init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "textcnnBase init failed, ret=" << ret << ".";
        return ret;
    }

    std::string inferPath = argv[1];
    std::vector<std::string> files;
    ret = read_files_from_path(inferPath + "ids", &files);
    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        return ret;
    }
    // do eval and calc the f1 score
    bool eval = atoi(argv[2]);
    for (uint32_t i = 0; i < files.size(); i++) {
        LogInfo << "==============================================================";
        LogInfo << "read file name: " << files[i];
        ret = textcnnBase->process(inferPath, files[i], eval);
        if (ret != APP_ERR_OK) {
            LogError << "textcnnBase process failed, ret=" << ret << ".";
            textcnnBase->de_init();
            return ret;
        }
    }

    if (eval) {
        LogInfo << "==============================================================";
        if (g_infer_cost.size()) {
            float accuracy = 1 - (g_fp + g_fn) * 1.0 / g_infer_cost.size();
            LogInfo << "Accuracy: " << accuracy;
        }
        if (g_tp + g_fp) {
            float precision = g_tp * 1.0 / (g_tp + g_fp);
            LogInfo << "Precision: " << precision;
            if (g_tp + g_fn) {
                float recall = g_tp * 1.0 / (g_tp + g_fn);
                LogInfo << "Recall: " << recall;
                if (precision + recall) {
                    LogInfo << "F1 Score: " << 2 * precision * recall / (precision + recall);
                    LogInfo << "==========================================================";
                }
            }
        }
    }

    textcnnBase->de_init();
    double costSum = 0;
    for (uint32_t i = 0; i < g_infer_cost.size(); i++) {
        costSum += g_infer_cost[i];
    }
    LogInfo << "Infer images sum " << g_infer_cost.size() << ", cost total time: " << costSum
            << " ms.";
    LogInfo << "The throughput: " << g_infer_cost.size() * 1000 / costSum << " bin/sec.";
    return APP_ERR_OK;
}

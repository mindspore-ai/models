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

#include "MxNCFBase.h"
#include <dirent.h>
#include <algorithm>
#include <numeric>
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;
double hr_total = 0.0;
uint32_t hr_num = 0;
double ndcgr_total = 0.0;
uint32_t ndcgr_num = 0;

void InitNCFParam(InitParam *initParam) { initParam->deviceId = 0; }

APP_ERROR ReadFilesFromPath(const std::string &path, std::vector<std::string> *files) {
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
        LogWarn << "Please input path, such as '../data/model/ncf.om'.";
        return APP_ERR_OK;
    }

    std::string modelPath = argv[1];
    InitParam initParam;
    InitNCFParam(&initParam);
    initParam.modelPath = modelPath;
    auto ncfBase = std::make_shared<NCFBase>();
    APP_ERROR ret = ncfBase->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "NCFBase init failed, ret=" << ret << ".";
        return ret;
    }

    std::string inferPath = argv[2];
    std::vector<std::string> files;
    ret = ReadFilesFromPath(inferPath + "tensor_0", &files);
    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        ncfBase->DeInit();
        return ret;
    }

    // infer and calc the HR & NDCG score
    bool eval = atoi(argv[3]);
    for (uint32_t i = 0; i < files.size(); i++) {
        LogInfo << "read file name: " << files[i];
        ret = ncfBase->Process(inferPath, files[i], eval);
        if (ret != APP_ERR_OK) {
            LogError << "NCFBase process failed, ret=" << ret << ".";
            ncfBase->DeInit();
            return ret;
        }
    }

    if (eval) {
        LogInfo << "==============================================================";
        if (hr_num == 0 || ndcgr_num == 0) {
            LogInfo << "Invalid divider， hr_num or ndcgr_num can not be 0";
        } else {
            double hr = hr_total / hr_num;
            double ndcgr = ndcgr_total / ndcgr_num;
            LogInfo << "average HR = " << hr << ", average NDCG = " << ndcgr;
        }
        LogInfo << "==============================================================";
    }

    ncfBase->DeInit();
    double costSum = std::accumulate(g_inferCost.begin(), g_inferCost.end(), 0.0);
    LogInfo << "Infer " << g_inferCost.size() << " batches, cost total time: " << costSum << " ms.";
    LogInfo << "Cost " << costSum / (1000 * g_inferCost.size()) << " sec per batch.";
    return APP_ERR_OK;
}

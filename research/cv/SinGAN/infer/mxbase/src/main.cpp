/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "SinGAN.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_infer_cost;
void InitSinGANParam(InitParam* initParam, const std::string &model_path, const std::string &output_data_path) {
    initParam->deviceId = 0;
    initParam->modelPath = model_path;
    initParam->outputDataPath = output_data_path;
}

APP_ERROR ReadFilesFromPath(const std::string &path, std::vector<std::string> *files) {
    DIR *dir = NULL;
    struct dirent *ptr = NULL;
    int file_type = 8;
    if ((dir = opendir(path.c_str())) == NULL) {
        LogError << "Open dir error: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    while ((ptr = readdir(dir)) != NULL) {
        // d_type == 8 is file
        if (ptr->d_type == file_type) {
            files->push_back(ptr->d_name);
        }
    }
    closedir(dir);
    return APP_ERR_OK;
}


int main(int argc, char* argv[]) {
    LogInfo << "=======================================  !!!Parameters setting!!!" << \
               "========================================";
    std::string model_path = argv[1];
    LogInfo << "==========  loading model weights from: " << model_path;

    std::string input_data_path = argv[2];
    LogInfo << "==========  input data path = " << input_data_path;

    std::string output_data_path = argv[3];
    LogInfo << "==========  output data path = " << output_data_path << \
               " WARNING: please make sure that this folder is created in advance!!!";

    LogInfo << "========================================  !!!Parameters setting!!! " << \
               "========================================";
    InitParam initParam;
    InitSinGANParam(&initParam, model_path, output_data_path);
    auto singan = std::make_shared<SinGAN>();
    APP_ERROR ret = singan->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "SinGAN init failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<std::string> files;
    ret = ReadFilesFromPath(input_data_path, &files);

    if (ret != APP_ERR_OK) {
        LogError << "Read files from path failed, ret=" << ret << ".";
        return ret;
    }
    // do infer

    ret = singan->Process(output_data_path, input_data_path, files[0], files[1]);
    if (ret != APP_ERR_OK) {
        LogError << "SinGAN process failed, ret=" << ret << ".";
        singan->DeInit();
        return ret;
    }

    LogInfo << "infer succeed and write the result data with binary file !";
    singan->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_infer_cost.size(); i++) {
        costSum += g_infer_cost[i];
    }
    LogInfo << "Infer images sum " << g_infer_cost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_infer_cost.size() * 1000 / costSum << " bin/sec.";
    LogInfo << "==========  The infer result has been saved in ---> " << output_data_path;
    return APP_ERR_OK;
}

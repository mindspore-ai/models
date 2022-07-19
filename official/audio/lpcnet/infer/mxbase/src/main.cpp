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
#include <sys/io.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include "MxBase/Log/Log.h"
#include "Lpcnet.h"

namespace {
    const int LOOP = 2620;
}  // namespace

APP_ERROR ReadTxt(const std::string &path, std::vector<std::vector<float>> &cfeat,
                  std::vector<std::vector<int>> &period, std::vector<std::vector<float>> &feature) {
    std::string cfeatPath = path + "_cfeat.txt";
    std::string periodPath = path + "_period.txt";
    std::string featurePath = path + "_feature.txt";
    std::ifstream inFile;
    float x;
    inFile.open(cfeatPath);
    if (!inFile) {
        LogError << "Unable to open file";
        exit(1);  // terminate with error
    }
    std::vector<float> tmp;
    int idx = 0;
    while (inFile >> x) {
        tmp.push_back(x);
        idx++;
        if (idx % 20 == 0) {
            cfeat.push_back(tmp);
            tmp.erase(tmp.begin(), tmp.end());
        }
    }
    inFile.close();

    inFile.open(periodPath);
    if (!inFile) {
        LogError << "Unable to open file";
        exit(1);  // terminate with error
    }
    while (inFile >> x) {
        std::vector<int>tmp2{static_cast<int>(x)};
        period.push_back(tmp2);
    }
    inFile.close();

    inFile.open(featurePath);
    if (!inFile) {
        LogError << "Unable to open file";
        exit(1);  // terminate with error
    }
    tmp.erase(tmp.begin(), tmp.end());
    idx = 0;
    while (inFile >> x) {
        tmp.push_back(x);
        idx++;
        if (idx % 36 == 0 && idx > 0) {
            feature.push_back(tmp);
            tmp.erase(tmp.begin(), tmp.end());
        }
    }
    inFile.close();
    return APP_ERR_OK;
}

void InitLpcnetParam(InitParam* initParam) {
    initParam->deviceId = 0;
    initParam->encoder_modelPath = "../data/enc.om";
    initParam->decoder_modelPath = "../data/dec.om";
    initParam->inferSrcTokensPath = "../data/testing-data/";
    initParam->resultName = "../result/mxbase/";
    initParam->begin = 0;
    initParam->end = LOOP;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        LogWarn << "Please input dataPath and saveName.";
        return APP_ERR_OK;
    }
    InitParam initParam;
    InitLpcnetParam(&initParam);
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.inferSrcTokensPath = argv[1];
    initParam.resultName = argv[2];
    initParam.begin = std::stoi(argv[3]);
    initParam.end = std::stoi(argv[4]);
    if (initParam.end > LOOP) {
        initParam.end = LOOP;
    }
    auto model_Lpcnet = std::make_shared<Lpcnet>();
    APP_ERROR ret = model_Lpcnet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Lpcnet init failed, ret=" << ret << ".";
        return ret;
    }
    for (int i = initParam.begin; i < initParam.end; i++) {
        std::string path = initParam.inferSrcTokensPath + std::to_string(i);
        std::vector<std::vector<float>> cfeat;
        std::vector<std::vector<int>> period;
        std::vector<std::vector<float>> feature;
        ret = ReadTxt(path, cfeat, period, feature);
        if (ret != APP_ERR_OK) {
            model_Lpcnet->DeInit();
            return ret;
        }
        std::vector<int16_t> mem_outputs;
        ret = model_Lpcnet->Process(cfeat, period, feature, initParam, mem_outputs);
        if (ret !=APP_ERR_OK) {
            LogError << "Lpcnet process failed, ret=" << ret << ".";
            model_Lpcnet->DeInit();
            return ret;
        }
        std::string save_path = initParam.resultName + std::to_string(i) + ".pcm";
        std::ofstream outfile(save_path);
        if (outfile.fail()) {
            LogError << "Failed to open result file: ";
            return APP_ERR_COMM_FAILURE;
        }
        for (int k = 0; k < mem_outputs.size(); k++) {
            outfile << mem_outputs[k] << " ";
        }
        outfile.close();
    }
    model_Lpcnet->DeInit();
    return APP_ERR_OK;
}

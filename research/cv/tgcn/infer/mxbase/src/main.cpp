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
#include <math.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstdio>
#include "Tgcn.h"
#include "MxBase/Log/Log.h"

APP_ERROR ReadAdj(const std::string &dataset, const std::string &adj_path, std::vector<std::vector<float>> *adj) {
    std::ifstream fp(adj_path);
    std::string line;
    int num = dataset == "SZ-taxi" ? 156 : 207;
    while (std::getline(fp, line)) {
        std::vector<float> data_adj;
        std::string number;
        std::istringstream readstr(line);
        for (int j = 0; j < num; j++) {
            std::getline(readstr, number, ' ');
            data_adj.emplace_back(atof(number.c_str()));
        }
        adj->emplace_back(data_adj);
    }
    return APP_ERR_OK;
}

APP_ERROR ReadFeat(const std::string &dataset, const std::string &feat_path, const int seq_len, const int pre_len,
                   float &max_val, std::vector<std::vector<float>> *feat_input,
                   std::vector<std::vector<float>> *feat_target) {
    std::ifstream fp(feat_path);
    std::string line;
    std::vector<std::vector<float>> feat;
    int num = dataset == "SZ-taxi" ? 156 : 207;
    int flag = 0;
    while (std::getline(fp, line)) {
        std::vector<float> data_feat;
        std::string number;
        std::istringstream readstr(line);
        for (int j = 0; j < num; j++) {
            std::getline(readstr, number, ',');
            float tmp = atof(number.c_str());
            max_val = std::max(max_val, tmp);
            data_feat.emplace_back(tmp);
        }
        if (!flag) {
            max_val = -1e9;
            flag = 1;
            continue;
        }
        feat.emplace_back(data_feat);
    }
    size_t time_len = feat.size();
    for (size_t i = 0 ; i < time_len - seq_len - pre_len; i++) {
        for (size_t j = i ; j < i + seq_len ; j++)
            feat_input->emplace_back(feat[j]);
        for (size_t j = i + seq_len ; j < i + seq_len + pre_len; j++)
            feat_target->emplace_back(feat[j]);
    }
    return APP_ERR_OK;
}

float Rmse(const std::vector<float> &output, const std::vector<float> &target) {
    float res = 0;
    size_t len = output.size();
    for (size_t i = 0; i < len; i++) {
        res += (output[i] - target[i]) * (output[i] - target[i]);
    }
    res = res / len;
    return sqrt(res);
}
float Mae(const std::vector<float> &output, const std::vector<float> &target) {
    float res = 0;
    size_t len = output.size();
    for (size_t i = 0; i < len; i++) {
        res += abs(output[i] - target[i]);
    }
    return res /= len;
}
float Acc(const std::vector<float> &output, const std::vector<float> &target) {
    float diff_norm = 0, targe_norm = 0;
    size_t len = output.size();
    for (size_t i = 0; i < len; i++) {
        diff_norm += (output[i] - target[i]) * (output[i] - target[i]);
        targe_norm += target[i] * target[i];
    }
    diff_norm = sqrt(diff_norm);
    targe_norm = sqrt(targe_norm);
    return 1 - diff_norm / targe_norm;
}
float R2(const std::vector<float> &output, const std::vector<float> &target) {
    float output_mean = 0, rsum = 0, rsumt = 0;
    size_t len = output.size();
    for (size_t i = 0; i < len; i++) {
        output_mean += output[i];
        rsum += (output[i] - target[i]) * (output[i] - target[i]);
    }
    output_mean /= len;
    for (size_t i = 0; i < len; i++)
        rsumt += (output_mean - target[i]) * (output_mean - target[i]);
    return 1 - rsum / rsumt;
}
float Var(const std::vector<float> &output, const std::vector<float> &target) {
    float diff_var = 0, target_var = 0, diff_mean = 0, target_mean = 0;
    std::vector<float> diff;
    size_t len = output.size();
    for (size_t i = 0; i < len; i++) {
        diff.emplace_back(target[i] - output[i]);
        diff_mean += target[i] - output[i];
        target_mean += target[i];
    }
    for (size_t i = 0; i < len; i++) {
        diff_var += (diff[i] - diff_mean) * (diff[i] - diff_mean);
        target_var += (target[i] - target_mean) * (target[i] - target_mean);
    }
    diff_var /= len;
    target_var /= len;
    return 1 - diff_var / target_var;
}

int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    std::string dataset = argv[1];
    initParam.dataset = dataset;
    std::string adj_path, feat_path;
    int seq_len, pre_len;
    if (dataset == "SZ-taxi") {
        initParam.modelPath = "../data/models/tgcn_sztaxi.om";
        adj_path = "../data/input/SZ-taxi/adj.csv";
        feat_path = "../data/input/SZ-taxi/feature.csv";
        seq_len = 4;
        pre_len = 1;
    }
    auto tgcn = std::make_shared<TGCN>();
    printf("Start running\n");
    APP_ERROR ret = tgcn->Init(initParam);
    if (ret != APP_ERR_OK) {
        tgcn->DeInit();
        LogError << "tgcn init failed, ret=" << ret << ".";
        return ret;
    }

    float max_val = -1e9;
    float rmse = 0, mae = 0, acc = 0, r2 = 0, var = 0;
    std::vector<std::vector<float>> adj_data, feat_input, feat_target;
    ret = ReadAdj(dataset, adj_path, &adj_data);
    if (ret != APP_ERR_OK) {
        tgcn->DeInit();
        LogError << "read ajd failed, ret=" << ret << ".";
        return ret;
    }
    ret = ReadFeat(dataset, feat_path, seq_len, pre_len, max_val, &feat_input, &feat_target);
    if (ret != APP_ERR_OK) {
        tgcn->DeInit();
        LogError << "read feat failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "Read data done.";

    // 输入归一化
    int feat_input_len = feat_input.size();
    int feat_input_content_len = feat_input[0].size();
    int feat_targe_len = feat_target.size();
    int feat_target_content_len = feat_target[0].size();
    for (int i = 0 ; i < feat_input_len ; i++)
        for (int j = 0 ; j < feat_input_content_len ; j++)
            feat_input[i][j] /= max_val;
    for (int i = 0 ; i < feat_targe_len ; i++)
        for (int j = 0 ; j < feat_target_content_len ; j++)
            feat_target[i][j] /= max_val;

    const int data_num = feat_input.size() / seq_len;

    std::vector<std::vector<float>> tot_output;
    int tot_num = 0;
    for (int i = data_num * 0.8 + 4; i < data_num ; i++) {
        tot_num++;
        std::vector<std::vector<float>> data;
        std::vector<float> target;
        std::vector<float> output;
        for (int j = 0 ; j < seq_len ; j++)
            data.push_back(feat_input[i * seq_len + j]);

        target.insert(target.end(), feat_target[i].begin(), feat_target[i].end());

        ret = tgcn->Process(dataset, data, initParam, &output);
        if (ret !=APP_ERR_OK) {
            LogError << "tgcn process failed, ret=" << ret << ".";
            tgcn->DeInit();
            return ret;
        }

        tot_output.push_back(output);
        rmse += (Rmse(output, target) * max_val);
        mae += (Mae(output, target) * max_val);
        acc += Acc(output, target);
        r2 += R2(output, target);
        var += Var(output, target);
    }
    LogInfo << "totla " << " rmse: " << rmse / (tot_num)
                << " mae: " << mae / (tot_num)
                << " acc: " << acc / (tot_num)
                << " r2: " <<  r2 / (tot_num)
                << " var: " << var / (tot_num);

    std::string resultPathName = "./result.txt";
    std::ofstream outfile(resultPathName, std::ios::out);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }
    for (auto u : tot_output) {
        std::string tmp;
        for (auto x : u) {
            tmp += std::to_string(x) + " ";
        }
        tmp = tmp.substr(0, tmp.size()-1);
        outfile << tmp << std::endl;
    }
    outfile.close();

    // tgcn->DeInit();
    return APP_ERR_OK;
}


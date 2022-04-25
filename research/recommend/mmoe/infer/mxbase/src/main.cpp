/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd
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
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstdio>
#include "MMoE.h"
#include "half.hpp"
#include "MxBase/Log/Log.h"

using half_float::half;

const char mode[] = "eval";
template<class dtype>
APP_ERROR ReadTxt(const std::string &path, std::vector<std::vector<dtype>> &dataset) {
    std::ifstream fp(path);
    std::string line;
    while (std::getline(fp, line)) {
        std::vector<dtype> data_line;
        std::string number;
        std::istringstream readstr(line);

        while (std::getline(readstr, number, '\t')) {
            data_line.push_back(half(atof(number.c_str())));
        }
        dataset.push_back(data_line);
    }
    return APP_ERR_OK;
}

APP_ERROR WriteResult(const std::string &output_dir, const std::string &filename,
                        const std::vector<std::vector<half>> &result) {
    std::string output_path = output_dir + "/" + filename;
    if (access(output_dir.c_str(), F_OK) == -1) {
        mkdir(output_dir.c_str(), S_IRWXO|S_IRWXG|S_IRWXU);
    }
    std::ofstream outfile(output_path, std::ios::out | std::ios::trunc);\
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }
    for (size_t i = 0; i < result.size(); i ++) {
        std::string temp = std::to_string(result[i][0]) + "\t" +std::to_string(result[i][1]) + "\n";
        outfile << temp;
    }
    outfile.close();
    return APP_ERR_OK;
}

float get_auc(const std::vector<std::vector<half>> &preds, const std::vector<std::vector<half>> &labels
                 , size_t n_bins = 1000000) {
    std::vector<half> flatten_preds;
    std::vector<half> flatten_labels;
    int rows = preds.size();
    for (size_t i = 0; i < rows; i ++) {
        flatten_preds.push_back(preds[i][0]);
        flatten_preds.push_back(preds[i][1]);
        flatten_labels.push_back(labels[i][0]);
        flatten_labels.push_back(labels[i][1]);
    }
    size_t positive_len = 0;
    for (size_t i = 0; i < flatten_labels.size(); i++) {
        positive_len += static_cast<int>(flatten_labels[i]);
    }
    size_t negative_len = flatten_labels.size()-positive_len;
    if (positive_len == 0 || negative_len == 0) {
        return 0.0;
    }
    uint64_t total_case = positive_len*negative_len;
    std::vector<size_t> pos_histogram(n_bins+1, 0);
    std::vector<size_t> neg_histogram(n_bins+1, 0);
    float bin_width = 1.0/n_bins;
    for (size_t i = 0; i < flatten_preds.size(); i ++) {
        size_t nth_bin = static_cast<int>(flatten_preds[i]/bin_width);
        if (static_cast<int>(flatten_labels[i]) == 1) {
            pos_histogram[nth_bin] += 1;
        } else {
            neg_histogram[nth_bin] += 1;
        }
    }
    size_t accumulated_neg = 0;
    float satisfied_pair = 0;
    for (size_t i = 0; i < n_bins+1; i ++) {
        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5);
        accumulated_neg += neg_histogram[i];
    }
    return satisfied_pair/total_case;
}

int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/MMoE.om";
    auto mmoe = std::make_shared<MMoE>();
    printf("Start running\n");
    APP_ERROR ret = mmoe->Init(initParam);
    if (ret != APP_ERR_OK) {
        mmoe->DeInit();
        LogError << "mmoe init failed, ret=" << ret << ".";
        return ret;
    }

    // read data from txt
    std::string data_path = "../data/input/data_" + std::string(mode) + std::string(".txt");
    std::string income_path = "../data/input/income_labels_" + std::string(mode) +std::string(".txt");
    std::string married_path = "../data/input/married_labels_" + std::string(mode) +std::string(".txt");
    std::vector<std::vector<half>> data;
    std::vector<std::vector<half>> income;
    std::vector<std::vector<half>> married;
    ret = ReadTxt(data_path, data);
    if (ret != APP_ERR_OK) {
        LogError << "read ids failed, ret=" << ret << ".";
        return ret;
    }

    ret = ReadTxt(income_path, income);
    if (ret != APP_ERR_OK) {
        LogError << "read wts failed, ret=" << ret << ".";
        return ret;
    }

    ret = ReadTxt(married_path, married);
    if (ret != APP_ERR_OK) {
        LogError << "read label failed, ret=" << ret << ".";
        return ret;
    }

    int data_rows = data.size();
    int income_rows = income.size();
    int married_rows = married.size();
    if (data_rows != income_rows || income_rows != married_rows) {
        LogError << "size of data, income and married are not equal";
        return -1;
    }
    int rows = data_rows;
    std::vector<std::vector<half>> income_preds;
    std::vector<std::vector<half>> married_preds;

    for (int i = 0; i < rows; i++) {
        std::vector<std::vector<half>> data_batch;
        data_batch.emplace_back(data[i]);
        ret = mmoe->Process(data_batch, initParam, income_preds, married_preds);
        if (ret !=APP_ERR_OK) {
            LogError << "mmoe process failed, ret=" << ret << ".";
            mmoe->DeInit();
            return ret;
        }
    }

    // write results
    std::string output_dir = "./output";
    std::string filename = "income_preds_" + std::string(mode) + std::string(".txt");
    WriteResult(output_dir, filename, income_preds);
    filename = "income_labels_" + std::string(mode) +std::string(".txt");
    WriteResult(output_dir, filename, income);
    filename = "married_preds_" + std::string(mode) + std::string(".txt");
    WriteResult(output_dir, filename, married_preds);
    filename = "married_labels_" + std::string(mode) +std::string(".txt");
    WriteResult(output_dir, filename, married);

    float infer_total_time = mmoe->GetInferCostMilliSec()/1000;
    float income_auc = get_auc(income_preds, income);
    float married_auc = get_auc(married_preds, married);
    LogInfo << "<<==========Infer Metric==========>>";
    LogInfo << "Number of samples:" + std::to_string(rows);
    LogInfo << "Total inference time:" + std::to_string(infer_total_time);
    LogInfo << "Average infer time:" + std::to_string(infer_total_time/rows);
    LogInfo << "Income infer auc:"+ std::to_string(income_auc);
    LogInfo << "Married infer auc:"+ std::to_string(married_auc);
    LogInfo << "<<================================>>";

    mmoe->DeInit();
    return APP_ERR_OK;
}

/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
#include "Autodis.h"
#include "MxBase/Log/Log.h"

const char input_format[] = "bin";
const int shape[3] = {39, 39, 1};
template<class dtype>
APP_ERROR ReadTxt(const std::string &path, std::vector<std::vector<dtype>> &dataset) {
    std::ifstream fp(path);
    std::string line;
    while (std::getline(fp, line)) {
        std::vector<dtype> data_line;
        std::string number;
        std::istringstream readstr(line);

        while (std::getline(readstr, number, '\t')) {
            data_line.push_back(atof(number.c_str()));
        }
        dataset.push_back(data_line);
    }
    return APP_ERR_OK;
}

template<class dtype>
APP_ERROR ReadBin(const std::string &path, std::vector<std::vector<dtype>> &dataset, const int index) {
    std::ifstream inFile(path, std::ios::in|std::ios::binary);
    dtype data[shape[index]];
    while (inFile.read(reinterpret_cast<char *>(&data), sizeof(data))) {
        std::vector<dtype> temp(data, data + sizeof(data) / sizeof(data[0]));
        dataset.push_back(temp);
    }
    return APP_ERR_OK;
}

APP_ERROR WriteResult(const std::string &output_dir, const std::string &output_file,
                        const std::vector<std::vector<float>> &label, const std::vector<float> &probs,
                        const std::vector<int> &pred) {
    std::string output_path = output_dir+"/"+output_file;
    if (access(output_dir.c_str(), F_OK) == -1) {
        mkdir(output_dir.c_str(), S_IRWXO|S_IRWXG|S_IRWXU);
    }
    std::ofstream outfile(output_path, std::ios::out | std::ios::trunc);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }
    outfile << "label\tprob\tpred\n";

    for (size_t i = 0; i < label.size(); i ++) {
        std::string temp = "";
        for (size_t j = 0; j < label[i].size(); j ++) {
            temp += std::to_string(static_cast<int>(label[i][j]))+"\t";
        }
        temp += std::to_string(probs[i])+"\t";
        temp += std::to_string(pred[i])+"\n";
        outfile << temp;
    }
    outfile.close();
    return APP_ERR_OK;
}

APP_ERROR WriteMetric(const std::string &output_dir, const std::string &output_file,
                        const int &data_row, const float &infer_total_time,
                        const float &acc, const float &auc) {
    std::string output_path = output_dir+"/"+output_file;
    if (access(output_dir.c_str(), F_OK) == -1) {
        mkdir(output_dir.c_str(), S_IRWXO | S_IRWXG | S_IRWXU);
    }
    std::ofstream outfile(output_path, std::ios::out | std::ios::trunc);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }
    outfile << "Number of samples:" + std::to_string(data_row) + "\n";
    outfile << "Infer total time:" + std::to_string(infer_total_time) + "\n";
    outfile << "Average infer time:" + std::to_string(infer_total_time/data_row) + "\n";
    outfile << "Infer acc:" + std::to_string(acc) + "\n";
    outfile << "Infer auc:" + std::to_string(auc) + "\n";
    return APP_ERR_OK;
}

float get_acc(const std::vector<int> &pred, const std::vector<std::vector<float>> &label) {
    size_t num = pred.size();
    if (num == 0) {
        return 0.0;
    }
    float s = 0;
    for (size_t i = 0; i < num; i++) {
        if (pred[i] == static_cast<int>(label[i][0])) {
            s += 1;
        }
    }
    return s/num;
}

float get_auc(const std::vector<float> &probs, const std::vector<std::vector<float>> &label, size_t n_bins = 10000) {
    size_t positive_len = 0;
    for (size_t i = 0; i < label.size(); i++) {
        positive_len += static_cast<int>(label[i][0]);
    }
    size_t negative_len = label.size()-positive_len;
    if (positive_len == 0 || negative_len == 0) {
        return 0.0;
    }
    uint64_t total_case = positive_len*negative_len;
    std::vector<size_t> pos_histogram(n_bins+1, 0);
    std::vector<size_t> neg_histogram(n_bins+1, 0);
    float bin_width = 1.0/n_bins;
    for (size_t i = 0; i < probs.size(); i ++) {
        size_t nth_bin = static_cast<int>(probs[i]/bin_width);
        if (static_cast<int>(label[i][0]) == 1) {
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
    initParam.modelPath = "../data/model/autodis.om";
    auto autodis = std::make_shared<AUTODIS>();
    printf("Start running\n");
    APP_ERROR ret = autodis->Init(initParam);
    if (ret != APP_ERR_OK) {
        autodis->DeInit();
        LogError << "autodis init failed, ret=" << ret << ".";
        return ret;
    }

    // read data from txt or bin
    std::vector<std::vector<int>> ids_data;
    std::vector<std::vector<float>> wts_data;
    std::vector<std::vector<float>> label_data;
    if (strcmp(input_format, "txt") == 0) {
        std::string ids_path = "../data/input/ids.txt";
        std::string wts_path = "../data/input/wts.txt";
        std::string label_path = "../data/input/label.txt";

        ret = ReadTxt(ids_path, ids_data);
        if (ret != APP_ERR_OK) {
            LogError << "read ids failed, ret=" << ret << ".";
            return ret;
        }
        ret = ReadTxt(wts_path, wts_data);
        if (ret != APP_ERR_OK) {
            LogError << "read wts failed, ret=" << ret << ".";
            return ret;
        }
        ret = ReadTxt(label_path, label_data);
        if (ret != APP_ERR_OK) {
            LogError << "read label failed, ret=" << ret << ".";
            return ret;
        }
    } else {
        std::string ids_path = "../data/input/ids.bin";
        std::string wts_path = "../data/input/wts.bin";
        std::string label_path = "../data/input/label.bin";

        ret = ReadBin(ids_path, ids_data, 0);
        if (ret != APP_ERR_OK) {
            LogError << "read ids failed, ret=" << ret << ".";
            return ret;
        }
        ret = ReadBin(wts_path, wts_data, 1);
        if (ret != APP_ERR_OK) {
            LogError << "read wts failed, ret=" << ret << ".";
            return ret;
        }
        ret = ReadBin(label_path, label_data, 2);
        if (ret != APP_ERR_OK) {
            LogError << "read label failed, ret=" << ret << ".";
            return ret;
        }
    }

    int ids_row = ids_data.size();
    int wts_row = wts_data.size();
    int label_row = label_data.size();

    if (label_row != ids_row || label_row != wts_row) {
        LogError << "size of label, ids and wts are not equal";
        return -1;
    }
    int data_row = label_row;

    std::vector<int> pred;
    std::vector<float> probs;

    for (int i = 0; i < data_row; i++) {
        std::vector<std::vector<int>> ids;
        std::vector<std::vector<float>> wts;
        std::vector<std::vector<float>> label;
        ids.emplace_back(ids_data[i]);
        wts.emplace_back(wts_data[i]);
        label.emplace_back(label_data[i]);
        ret = autodis->Process(ids, wts, label, initParam, pred, probs);
        if (ret !=APP_ERR_OK) {
            LogError << "autodis process failed, ret=" << ret << ".";
            autodis->DeInit();
            return ret;
        }
    }

    std::string output_dir = "./output";
    std::string output_file = "result.txt";
    WriteResult(output_dir, output_file, label_data, probs, pred);

    float infer_total_time = autodis->GetInferCostMilliSec()/1000;
    float acc = get_acc(pred, label_data);
    float auc = get_auc(probs, label_data);
    output_file = "metric.txt";
    WriteMetric(output_dir, output_file, data_row, infer_total_time, acc, auc);

    LogInfo << "<<==========Infer Metric==========>>";
    LogInfo << "Number of samples:" + std::to_string(data_row);
    LogInfo << "Infer total time:" + std::to_string(infer_total_time);
    LogInfo << "Average infer time:" + std::to_string(infer_total_time/data_row);
    LogInfo << "Infer acc:"+ std::to_string(acc);
    LogInfo << "Infer auc:"+ std::to_string(auc);
    LogInfo << "<<================================>>";

    autodis->DeInit();
    return APP_ERR_OK;
}

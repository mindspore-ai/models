/*
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
#include "AlignedReID.h"
#include <set>
#include <map>
#include <regex>
#include <cstring>
#include <cstdio>
#include <string>
#include <cmath>
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NUM = 751;
}
std::map<int, int> mapping();


int infer(const std::string& argv1, const std::string& argv2, const std::string& argv3, const std::string& argv4);
std::vector<std::vector<float> > calculate_distmat();
std::vector<int> query_pids;
std::vector<int> gallery_pids;
unsigned query_num = 0;
unsigned gallery_num = 0;
const unsigned FEATURE_SIZE = 2048;
std::vector<std::vector<float> > qf(3500);
std::vector<std::vector<float> > gf(20000);


int main(int argc, char *argv[]) {
    LogInfo << "====== parameters setting =======";
    std::string model_path = argv[1];
    LogInfo << "====== loading model from:  " << model_path;
    std::string result_file = argv[2];
    LogInfo << "====== result_file path:  " << result_file;
    std::string gallery_path = argv[3];
    LogInfo << "====== gallery_path path:  " << gallery_path;
    std::string query_path = argv[4];
    LogInfo << "====== query_path path:  " << query_path;
    std::vector<std::vector<float> > distmat;

    int ret = infer(model_path, query_path, result_file, gallery_path);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    LogInfo << "==================================";
    LogInfo << "Inference success, qf.size = " << query_num << "  gf.size = " << gallery_num;
    LogInfo << "==================================";

    distmat = calculate_distmat();

    LogInfo << "==================================";
    LogInfo << " All data has been loaded";
    LogInfo << " query_size = " << query_pids.size();
    LogInfo << " gallery_size = " << gallery_pids.size();
    LogInfo << "==================================";

    std::ofstream fout(result_file);
    for (unsigned i = 0; i < query_num; i++) {
        float min_element = 1000;
        unsigned min_index = -1;
        for (unsigned j = 0; j < gallery_num; j++) {
            min_element = std::min(min_element, distmat[i][j]);
            if (distmat[i][j] == min_element)
                min_index = j;
        }

        int query_pid = query_pids[i];
        int inferred_pid = gallery_pids[min_index];

        std::cout << "  query_pid: " << query_pid;
        std::cout.width(8 - static_cast<int>(log10(query_pid) + 1));
        std::cout << " -> " << " inferred_pid: " << inferred_pid;
        std::cout.width(8 - static_cast<int>(log10(inferred_pid) + 1));
        std::cout << "   img_index: " << min_index << "\n";

        fout << "query pid: " << query_pid << "  inferred pid: " << inferred_pid << "\n";
    }
    fout.close();
    LogInfo << "==================================";
}

std::vector<std::vector<float> > calculate_distmat() {
    unsigned m = query_num;
    unsigned n = gallery_num;
    LogInfo << "==================================";
    LogInfo << "qf.size = " << m;
    LogInfo << "gf.size = " << n;
    LogInfo << " qf normalization......";
    for (unsigned i = 0; i < m; i++) {
        float sum = 0;
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            sum += pow(qf[i][j], 2);
        float sqt = sqrt(sum);
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            qf[i][j] /= (sqt + 1);
    }
    LogInfo << " gf normalization......";
    // gf normalization
    for (unsigned i = 0; i < n; i++) {
        float sum = 0;
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            sum += pow(gf[i][j], 2);
        float sqt = sqrt(sum);
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            gf[i][j] /= (sqt + 1);
    }
    std::vector<float> a(m);
    for (unsigned i = 0; i < m; i++) {
        a[i] = 0;
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            a[i] += pow(qf[i][j], 2);
    }
    std::vector<float> b(n);
    for (unsigned i = 0; i < n; i++) {
        b[i] = 0;
        for (unsigned j = 0; j < FEATURE_SIZE; j++)
            b[i] += pow(gf[i][j], 2);
    }

    std::vector<std::vector<float> > t1(m);
    std::vector<std::vector<float> > t2(m);
    for (unsigned i = 0; i < m; i++) {
        for (unsigned j = 0; j < n; j++) {
            t1[i].push_back(a[i]);
            t2[i].push_back(b[j]);
        }
    }

    std::vector<std::vector<float> > distmat(m);
    LogInfo << " Doing distmat addition......";
    for (unsigned i = 0; i < m; i++) {
        for (unsigned j = 0; j < n; j++) {
            distmat[i].push_back(0);
            distmat[i][j] = t1[i][j] + t2[i][j];
        }
    }

    LogInfo << " Doing distmat multiplication......";

    for (unsigned i = 0; i < m; i++) {
        if (i % 336 == 0)
            std::cout << " doing distmat multiplication ----------------->\n";
        for (unsigned j = 0; j < n; j++) {
            float num = 0;
            for (unsigned k = 0; k < FEATURE_SIZE; k++)
                num += qf[i][k] * gf[j][k];
            distmat[i][j] += (-2) * num;
        }
    }
    std::cout << " doing operations have been done ---------------\n";

    LogInfo << "==================================";
    return distmat;
}

int infer(const std::string& argv1, const std::string& argv2, const std::string& argv3, const std::string& argv4) {
    std::ofstream fout(argv3);
    fout.close();

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;

    initParam.iou_thresh = 0.6;
    initParam.score_thresh = 0.6;
    initParam.checkTensor = true;

    initParam.modelPath = argv1;
    auto AlignedReID1 = std::make_shared<AlignedReID>();
    APP_ERROR ret = AlignedReID1->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "AlignedReID init failed, ret=" << ret << ".";
        return ret;
    }
    std::regex pattern("([-\\d]+)");

    std::string PATH = argv2;
    struct dirent *ptr;
    DIR *dir = opendir(PATH.c_str());
    std::vector<std::string> files;
    while ((ptr = readdir(dir)) != NULL)
        files.push_back(ptr->d_name);

    std::smatch result;
    LogInfo << "==================================";
    LogInfo << "Getting query features......";
    LogInfo << "Number of query images = " << files.size();
    for (unsigned int i = 0; i < files.size(); ++i) {
        int pos = files[i].find_last_of('.');
        std::string suffix(files[i].substr(pos + 1));
        if (suffix != "jpg")
            continue;
        std::string::const_iterator iterStart = files[i].begin();
        std::string::const_iterator iterEnd = files[i].end();
        if (regex_search(iterStart, iterEnd, result, pattern)) {
            int pid = atoi(std::string(result[0]).c_str());
            query_pids.push_back(pid);
        }
        /**********************/
        std::string file_name = PATH + files[i];
        ret = AlignedReID1->Process(file_name, qf[query_num]);

        if (ret != APP_ERR_OK) {
            LogError << "AlignedReID process failed, ret=" << ret << ".";
            AlignedReID1->DeInit();
            return ret;
        }
        query_num++;
    }
    LogInfo << " 100% have done ";
    LogInfo << "Extracted features from query set, obtained " << query_num << "-by-" << qf[0].size() << "matrix";
    LogInfo << "==================================";

    PATH = argv4;
    dir = opendir(PATH.c_str());
    files.clear();
    while ((ptr = readdir(dir)) != NULL)
        files.push_back(ptr->d_name);

    std::smatch result_;
    LogInfo << "==================================";
    LogInfo << "Getting gallery features......";
    LogInfo << "Number of gallery images = " << files.size();
    for (unsigned int i = 0; i < files.size(); ++i) {
        int pos = files[i].find_last_of('.');
        std::string suffix(files[i].substr(pos + 1));
        if (suffix != "jpg")
            continue;
        std::string::const_iterator iterStart = files[i].begin();
        std::string::const_iterator iterEnd = files[i].end();
        if (regex_search(iterStart, iterEnd, result_, pattern)) {
            int pid = atoi(std::string(result_[0]).c_str());
            if (pid < 0 || pid > 1501)
                continue;
            gallery_pids.push_back(pid);
        }
        std::string file_name = PATH + files[i];
        ret = AlignedReID1->Process(file_name, gf[gallery_num]);
        if (ret != APP_ERR_OK) {
            LogError << "AlignedReID process failed, ret=" << ret << ".";
            AlignedReID1->DeInit();
            return ret;
        }
        gallery_num++;
    }
    LogInfo << " 100% have done ";
    LogInfo << "Extracted features from gallery set, obtained " << gallery_num << "-by-" << gf[0].size() << "matrix";
    LogInfo << "==================================";

    AlignedReID1->DeInit();
    return APP_ERR_OK;
}

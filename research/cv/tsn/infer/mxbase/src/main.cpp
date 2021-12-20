/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>
#include "tsnClassification.h"
#include "MxBase/Log/Log.h"

APP_ERROR ReadList(const std::string &path, std::vector<std::string> *root_path, std::vector<int> *num_frames) {
    std::ifstream infile;
    infile.open(path.data());
    std::string s;
    while (std::getline(infile, s)) {
        std::istringstream is(s);
        int num_frame;
        std::string str;
        is >> str;
        is >> num_frame;
        root_path->push_back(str);
        num_frames->push_back(num_frame);
    }
    infile.close();
    return APP_ERR_OK;
}


int main(int argc, char* argv[]) {
    if (argc <= 3) {
        LogWarn << "Please input dataset path and n_pred, such as './data/vel/csv 9'";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/tsn_rgb.om";
    auto tsn = std::make_shared<TSN>();

    std::string dataPath = argv[1];
    std::string listPath = argv[2];
    std::string modality = argv[3];

    std::vector<std::string> root_path;
    std::vector<int> num_frames;

    auto ret = ReadList(listPath, &root_path, &num_frames);
    if (ret != APP_ERR_OK) {
        tsn->DeInit();
        LogError << "read dataset failed, ret=" << ret << ".";
        return ret;
    }
    int length = 3;
    int test_crops = 10;
    int data_length = 5;
    int input_size = 224;
    int test_segments = 25;

    if (modality == "RGB") {
        length = 3;
        data_length = 1;
    } else if (modality == "Flow") {
        length = 10;
    } else if (modality == "RGBDiff") {
        length = 18;
    }
    int scale_size = input_size * 256 / 224;
    std::vector<int> input_mean = {104, 117, 128};
    std::vector<int> input_std = {1, 1, 1};

    if (modality == "Flow") {
        input_mean = {128};
        input_std = {1};
    } else if (modality == "RGBDiff") {
        std::transform(std::begin(input_mean), std::end(input_mean),
         std::begin(input_mean), [data_length](int tmp){ return tmp * (1 + data_length);});
    }

    std::string image_tmpl;
    if (modality == "RGB" || modality == "RGBDiff") {
        image_tmpl = "img_";
    } else {
        image_tmpl = "flow_";
    }
    initParam.input_mean = input_mean;
    initParam.input_size = input_size;
    initParam.scale_size = scale_size;
    initParam.input_std = input_std;
    initParam.data_path = dataPath;
    initParam.modality = modality;
    initParam.length = length;
    initParam.test_crops = test_crops;

    ret = tsn->Init(initParam);
    if (ret != APP_ERR_OK) {
        tsn->DeInit();
        LogError << "tsn init failed, ret=" << ret << ".";
        return ret;
    }
    for (uint32_t i = 0; i < root_path.size(); ++i) {
        double tick = (num_frames[i] - data_length + 1) / static_cast<double>(test_segments);
        std::vector<int> segment_indices;
        std::vector<int> indices;
        for (int j = 0; j < test_segments; ++j) {
            double frame = tick / 2.0 + tick * j;
            segment_indices.push_back(static_cast<int>(frame) + 1);
        }
        for (auto seg_ind : segment_indices) {
            int p = static_cast<int>(seg_ind);
            for (int k = 0; k < data_length; ++k) {
                indices.push_back(p);
                if (p < num_frames[i]) {
                    ++p;
                }
            }
        }

        ret = tsn->Process(root_path[i], image_tmpl, indices, initParam);
        if (ret != APP_ERR_OK) {
            LogError << "tsn process failed, ret=" << ret << ".";
            tsn->DeInit();
            return ret;
        }
    }
    tsn->DeInit();
    return APP_ERR_OK;
}


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

#include <sys/time.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "../inc/utils.h"
#include "include/dataset/execute.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision.h"
#include "include/dataset/vision_ascend.h"
#include "include/api/types.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"


using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::dataset::Execute;
using mindspore::MSTensor;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::Graph;
using mindspore::dataset::Execute;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::Decode;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::HWC2CHW;

DEFINE_string(model_path, "../mcnn.mindir", "model path");
DEFINE_string(test_path, "../test_data/preprocess_data", "test dataset path");
DEFINE_string(query_path, "../test_data/preprocess_data", "query dataset path");
DEFINE_int32(input_width, 960, "input width");
DEFINE_int32(input_height, 576, "inputheight");
DEFINE_int32(device_id, 0, "device id");


int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_model_path).empty()) {
        std::cout << "Invalid mindir" << std::endl;
        return 1;
    }

    auto context = std::make_shared<Context>();
    auto ascend310_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(FLAGS_device_id);
    context->MutableDeviceInfo().push_back(ascend310_info);

    Graph graph;
    Status ret = Serialization::Load(FLAGS_model_path, ModelType::kMindIR, &graph);
    if (ret != kSuccess) {
        std::cout << "Load model failed." << std::endl;
        return 1;
    }

    Model model;
    ret = model.Build(GraphCell(graph), context);
    if (ret != kSuccess) {
        std::cout << "ERROR: Build failed." << std::endl;
        return 1;
    }

    std::vector<MSTensor> modelInputs = model.GetInputs();

    auto all_files = GetAllFiles(FLAGS_test_path);
    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }

    std::map<double, double> costTime_map;
    size_t size = all_files.size();

    for (size_t i = 0; i < size; ++i) {
        struct timeval start;
        struct timeval end;
        double startTime_ms;
        double endTime_ms;
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;

        std::cout << "Start predict input files:" << all_files[i] << std::endl;

        mindspore::MSTensor image =  ReadFileToTensor(all_files[i]);

        inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(),
                            image.Data().get(), image.DataSize());

        gettimeofday(&start, NULL);
        ret = model.Predict(inputs, &outputs);
        gettimeofday(&end, NULL);
        if (ret != kSuccess) {
            std::cout << "Predict " << all_files[i] << " failed." << std::endl;
            return 1;
        }
        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
        int rst = WriteResult(all_files[i], outputs);
        if (rst != 0) {
            std::cout << "write result failed." << std::endl;
            return rst;
        }
    }

    cal_time(costTime_map);

    auto all_files2 = GetAllFiles(FLAGS_query_path);
    if (all_files2.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }

    std::map<double, double> costTime_map2;
    size_t size2 = all_files2.size();

    for (size_t i = 0; i < size2; ++i) {
        struct timeval start;
        struct timeval end;
        double startTime_ms;
        double endTime_ms;
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;

        std::cout << "Start predict input files:" << all_files2[i] << std::endl;

        mindspore::MSTensor image =  ReadFileToTensor(all_files2[i]);

        inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(),
                            image.Data().get(), image.DataSize());

        gettimeofday(&start, NULL);
        ret = model.Predict(inputs, &outputs);
        gettimeofday(&end, NULL);
        if (ret != kSuccess) {
            std::cout << "Predict " << all_files2[i] << " failed." << std::endl;
            return 1;
        }
        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map2.insert(std::pair<double, double>(startTime_ms, endTime_ms));
        int rst = WriteResult(all_files2[i], outputs);
        if (rst != 0) {
            std::cout << "write result failed." << std::endl;
            return rst;
        }
    }
    cal_time(costTime_map2);

    return 0;
}

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

#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/dataset/vision_ascend.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"
#include "inc/utils.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::dataset::Execute;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::HWC2CHW;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::Decode;

DEFINE_string(gen_mindir_path, "", "generator mindir path");
DEFINE_string(dataset_path, "", "dataset path");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_gen_mindir_path).empty()) {
        std::cout << "Invalid generator mindir" << std::endl;
        return 1;
    }
    auto context = std::make_shared<Context>();
    auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310->SetDeviceID(FLAGS_device_id);
    context->MutableDeviceInfo().push_back(ascend310);

    mindspore::Graph gen_graph;
    Serialization::Load(FLAGS_gen_mindir_path, ModelType::kMindIR, &gen_graph);

    Model gen_model;
    Status gen_ret = gen_model.Build(GraphCell(gen_graph), context);

    if (gen_ret != kSuccess) {
        std::cout << "ERROR: Generator build failed." << std::endl;
        return 1;
    }

    std::vector<MSTensor> modelInputs = gen_model.GetInputs();
    if (modelInputs.empty()) {
        std::cout << "modelInputs empty." << std::endl;
        return 1;
    }

    auto all_data = GetAllFiles(FLAGS_dataset_path + "data");

    if (all_data.empty()) {
        std::cout << "files empty." << std::endl;
        return 1;
    }
    std::map<double, double> costTime_map;
    size_t size = all_data.size();

    for (size_t i = 0; i < size; ++i) {
        double startTimeMs;
        double endTimeMs;
        struct timeval start = {0};
        struct timeval end = {0};
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files:" << all_data[i] << std::endl;
        auto input = ReadFileToTensor(all_data[i]);

        inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(),
                            input.Data().get(), input.DataSize());

        gettimeofday(&start, nullptr);
        gen_ret = gen_model.Predict(inputs, &outputs);

        gettimeofday(&end, nullptr);
        if (gen_ret != kSuccess) {
            std::cout << "Generator inference " << all_data[i] << " failed." << std::endl;
            return 1;
        }

        WriteResult(all_data[i], outputs);
        startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
    }
    double average = 0.0;
    int inferCount = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        inferCount++;
    }
    average = average / inferCount;
    std::stringstream timeCost;
    timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << inferCount << std::endl;
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << std::endl;
    std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
    fileStream << timeCost.str();
    fileStream.close();
    costTime_map.clear();
    return 0;
}

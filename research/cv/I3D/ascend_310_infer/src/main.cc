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

#include <sys/stat.h>
#include <sys/time.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"
#include "../inc/utils.h"
#include "include/api/types.h"

namespace ms = mindspore;
namespace ds = mindspore::dataset;

DEFINE_string(mindir_path, "./i3d_minddir_64_rgb.mindir"
                           "i3d_minddir_64_rgb.mindir", "mindir path");
DEFINE_string(input0_path, "./data", "input0 path");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (RealPath(FLAGS_mindir_path).empty()) {
        std::cout << "Invalid mindir" << std::endl;
        return 1;
    }

    auto context = std::make_shared<ms::Context>();
    auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(FLAGS_device_id);
    context->MutableDeviceInfo().push_back(ascend310_info);

    // define model
    ms::Graph graph;
    ms::Status ret = ms::Serialization::Load(FLAGS_mindir_path, ms::ModelType::kMindIR, &graph);
    if (ret != ms::kSuccess) {
        std::cout << "ERROR00: Load model failed." << std::endl;
        return 1;
    }
    ms::Model I3D;

    // build model
    ret = I3D.Build(ms::GraphCell(graph), context);
    if (ret != ms::kSuccess) {
        std::cout << "ERROR00: Build model failed." << std::endl;
        return 1;
    }

    // get model info
    std::vector<ms::MSTensor> model_inputs = I3D.GetInputs();
    if (model_inputs.empty()) {
        std::cout << "ERROR00: Invalid model, inputs is empty." << std::endl;
        return 1;
    }
    // get input data info
    auto all_files = GetAllFiles(FLAGS_input0_path);
    if (all_files.empty()) {
        std::cout << "ERROR00: no input data." << std::endl;
        return 1;
    }

    std::map<double, double> costTime_map;
    std::vector<std::string> images = GetAllFiles(FLAGS_input0_path);
    for (const auto& image_file : images) {
        // prepare input
        struct timeval start = { 0 };
        struct timeval end = { 0 };
        double startTime_ms;
        double endTime_ms;
        std::vector<ms::MSTensor> outputs;
        std::vector<ms::MSTensor> inputs;
        std::cout << "Start predict input files:" << image_file << std::endl;
        // read image file and preprocess
        auto image = ReadFileToTensor(image_file);
        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            image.Data().get(), image.DataSize());
        // do eval
        std::cout << "start predict" << image_file << std::endl;
        gettimeofday(&start, NULL);
        ret = I3D.Predict(inputs, &outputs);
        gettimeofday(&end, NULL);
        if (ret != ms::kSuccess) {
            std::cout << "Predict " << image_file << " failed." << std::endl;
            return 1;
        }
        // time cost
        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
        WriteResult(image_file, outputs);
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

/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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


    auto all_files0 = GetAllFiles(FLAGS_dataset_path + "A");
    auto all_files1 = GetAllFiles(FLAGS_dataset_path + "bg_A");
    auto all_files2 = GetAllFiles(FLAGS_dataset_path + "eyel_A");
    auto all_files3 = GetAllFiles(FLAGS_dataset_path + "eyer_A");
    auto all_files4 = GetAllFiles(FLAGS_dataset_path + "nose_A");
    auto all_files5 = GetAllFiles(FLAGS_dataset_path + "mouth_A");
    auto all_files6 = GetAllFiles(FLAGS_dataset_path + "hair_A");
    auto all_files7 = GetAllFiles(FLAGS_dataset_path + "mask");
    auto all_files8 = GetAllFiles(FLAGS_dataset_path + "mask2");

    std::map<double, double> costTime_map;
    size_t size = all_files0.size();

    for (size_t i = 0; i < size; ++i) {
        double startTimeMs;
        double endTimeMs;
        struct timeval start = {0};
        struct timeval end = {0};
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files:" << all_files0[i] << std::endl;
        auto input0 = ReadFileToTensor(all_files0[i]);
        auto input1 = ReadFileToTensor(all_files1[i]);
        auto input2 = ReadFileToTensor(all_files2[i]);
        auto input3 = ReadFileToTensor(all_files3[i]);
        auto input4 = ReadFileToTensor(all_files4[i]);
        auto input5 = ReadFileToTensor(all_files5[i]);
        auto input6 = ReadFileToTensor(all_files6[i]);
        auto input7 = ReadFileToTensor(all_files7[i]);
        auto input8 = ReadFileToTensor(all_files8[i]);

        inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(),
                            input0.Data().get(), input0.DataSize());
        inputs.emplace_back(modelInputs[1].Name(), modelInputs[1].DataType(), modelInputs[1].Shape(),
                            input1.Data().get(), input1.DataSize());
        inputs.emplace_back(modelInputs[2].Name(), modelInputs[2].DataType(), modelInputs[2].Shape(),
                            input2.Data().get(), input2.DataSize());
        inputs.emplace_back(modelInputs[3].Name(), modelInputs[3].DataType(), modelInputs[3].Shape(),
                            input3.Data().get(), input3.DataSize());
        inputs.emplace_back(modelInputs[4].Name(), modelInputs[4].DataType(), modelInputs[4].Shape(),
                            input4.Data().get(), input4.DataSize());
        inputs.emplace_back(modelInputs[5].Name(), modelInputs[5].DataType(), modelInputs[5].Shape(),
                            input5.Data().get(), input5.DataSize());
        inputs.emplace_back(modelInputs[6].Name(), modelInputs[6].DataType(), modelInputs[6].Shape(),
                            input6.Data().get(), input6.DataSize());
        inputs.emplace_back(modelInputs[7].Name(), modelInputs[7].DataType(), modelInputs[7].Shape(),
                            input7.Data().get(), input7.DataSize());
        inputs.emplace_back(modelInputs[8].Name(), modelInputs[8].DataType(), modelInputs[8].Shape(),
                            input8.Data().get(), input8.DataSize());

        gettimeofday(&start, nullptr);
        gen_ret = gen_model.Predict(inputs, &outputs);

        gettimeofday(&end, nullptr);
        if (gen_ret != kSuccess) {
            std::cout << "Generator inference " << all_files0[i] << " failed." << std::endl;
            return 1;
        }

        WriteResult(all_files0[i], outputs);
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

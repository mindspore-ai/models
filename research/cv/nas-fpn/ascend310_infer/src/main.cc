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
using mindspore::DataType;
using mindspore::dataset::Execute;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::Pad;
using mindspore::dataset::vision::HWC2CHW;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::Decode;
using mindspore::dataset::InterpolationMode;

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(precision_mode, "allow_fp32_to_fp16", "precision mode");
DEFINE_string(op_select_impl_mode, "high_precision", "op impl mode");
DEFINE_string(buffer_optimize_mode, "off_optimize", "buffer optimize mode");
DEFINE_int32(image_height, 640, "image height");
DEFINE_int32(image_width, 640, "image width");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_mindir_path).empty()) {
        std::cout << "Invalid mindir" << std::endl;
        return 1;
    }

    auto context = std::make_shared<Context>();
    auto ascend310_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(FLAGS_device_id);
    ascend310_info->SetPrecisionMode(FLAGS_precision_mode);
    ascend310_info->SetOpSelectImplMode(FLAGS_op_select_impl_mode);
    ascend310_info->SetBufferOptimizeMode(FLAGS_buffer_optimize_mode);
    context->MutableDeviceInfo().push_back(ascend310_info);

    mindspore::Graph graph;
    Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);

    Model model;
    Status ret = model.Build(GraphCell(graph), context);
    if (ret != kSuccess) {
        std::cout << "ERROR: Build failed." << std::endl;
        return 1;
    }

    std::vector<MSTensor> modelInputs = model.GetInputs();
    if (modelInputs.empty()) {
        std::cout << "Invalid model, inputs is empty." << std::endl;
        return 1;
    }

    auto all_files = GetAllFiles(FLAGS_dataset_path);
    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }

    auto decode = Decode();
    auto resize = Resize({FLAGS_image_width, FLAGS_image_height});
    auto normalize = Normalize({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    auto hwc2chw = HWC2CHW();

    mindspore::dataset::Execute transform({decode, resize, normalize, hwc2chw});

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

        transform(image, &image);

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
        WriteResult(all_files[i], outputs);
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
    std::cout << "NN inference cost average time: "<< average << " ms of infer_count " << inferCount << std::endl;
    std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
    fileStream << timeCost.str();
    fileStream.close();
    costTime_map.clear();

    return 0;
}

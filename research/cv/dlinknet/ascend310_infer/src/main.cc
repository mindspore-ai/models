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

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // set context
    auto context = std::make_shared<ms::Context>();
    auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(FLAGS_device_id);
    context->MutableDeviceInfo().push_back(ascend310_info);

    // define model
    ms::Graph graph;
    ms::Status ret = ms::Serialization::Load(FLAGS_mindir_path, ms::ModelType::kMindIR, &graph);
    if (ret != ms::kSuccess) {
        std::cout << "Load model failed." << std::endl;
        return 1;
    }
    ms::Model dinknet34;

    // build model
    ret = dinknet34.Build(ms::GraphCell(graph), context);
    if (ret != ms::kSuccess) {
        std::cout << "Build model failed." << std::endl;
        return 1;
    }

    // get model info
    std::vector<ms::MSTensor> model_inputs = dinknet34.GetInputs();
    if (model_inputs.empty()) {
        std::cout << "Invalid model, inputs is empty." << std::endl;
        return 1;
    }
    std::map<double, double> costTime_map;
    // define transforms
    std::shared_ptr<ds::TensorTransform> decode(new ds::vision::Decode());
    std::shared_ptr<ds::TensorTransform> swapredblue(new ds::vision::SwapRedBlue());
    std::shared_ptr<ds::TensorTransform> normalize(new ds::vision::Normalize({103.53, 116.28, 123.675},
        {57.375, 57.120, 58.395}));
    std::shared_ptr<ds::TensorTransform> hwc2chw(new ds::vision::HWC2CHW());

    // define preprocessor
    ds::Execute preprocessor({decode, swapredblue, normalize, hwc2chw});

    std::vector<std::string> images = GetAllFiles(FLAGS_dataset_path);
    for (const auto& image_file : images) {
        // prepare input
        struct timeval start = { 0 };
        struct timeval end = { 0 };
        double startTime_ms;
        double endTime_ms;
        std::vector<ms::MSTensor> outputs;
        std::vector<ms::MSTensor> inputs;

        // read image file and preprocess
        auto image = ReadFileToTensor(image_file);
        ret = preprocessor(image, &image);
        if (ret != ms::kSuccess) {
            std::cout << "Image preprocess failed." << std::endl;
            return 1;
        }

        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
            image.Data().get(), image.DataSize());

        // infer
        gettimeofday(&start, NULL);
        ret = dinknet34.Predict(inputs, &outputs);
        gettimeofday(&end, NULL);
        if (ret != ms::kSuccess) {
            std::cout << "Predict model failed." << std::endl;
            return 1;
        }
        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
        // write infer result
        WriteResult(image_file, outputs);
        }
        double average = 0.0;
        int infer_cnt = 0;

        for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
            double diff = 0.0;
            diff = iter->second - iter->first;
            average += diff;
            infer_cnt++;
        }
        average = average / infer_cnt;
        std::stringstream timeCost;
        timeCost << "inference cost average time: " << average << " ms of infer_count " << infer_cnt << std::endl;
        std::cout << "inference cost average time: " << average << "ms of infer_count " << infer_cnt << std::endl;
        std::string file_name = "./time_Result" + std::string("/test_perform_static.txt");
        std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
        file_stream << timeCost.str();
        file_stream.close();
        costTime_map.clear();
        return 0;
}




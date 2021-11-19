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
#include <gflags/gflags.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "../inc/utils.h"
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/dataset/vision.h"
#include "include/dataset/transforms.h"
#include "include/dataset/execute.h"

namespace ms = mindspore;
namespace ds = mindspore::dataset;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::Pad;
using mindspore::dataset::vision::HWC2CHW;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::SwapRedBlue;
using mindspore::dataset::vision::Decode;
DEFINE_string(midas_file, "", "mindir path");
DEFINE_string(image_path, "", "dataset path");
DEFINE_string(dataset_name, "", "dataset name");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    auto context = std::make_shared<ms::Context>();
    auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(0);
    context->MutableDeviceInfo().push_back(ascend310_info);
    ms::Graph graph;
    std::cout << "midas file is " << FLAGS_midas_file << std::endl;
    ms::Status ret = ms::Serialization::Load(FLAGS_midas_file, ms::ModelType::kMindIR, &graph);
    if (ret != ms::kSuccess) {
        std::cout << "Load model failed." << std::endl;
        return 1;
    }
    ms::Model midas;
    ret = midas.Build(ms::GraphCell(graph), context);
    if (ret != ms::kSuccess) {
        std::cout << "Build model failed." << std::endl;
        return 1;
    }
    std::vector<ms::MSTensor> model_inputs = midas.GetInputs();
    if (model_inputs.empty()) {
        std::cout << "Invalid model, inputs is empty." << std::endl;
        return 1;
    }
    // Decode the input to YUV420 format
    std::shared_ptr<TensorTransform> decode(new Decode());
    // Resize the image to the given size
    std::shared_ptr<TensorTransform> resize(new Resize({384, 384}));
    std::shared_ptr<TensorTransform> normalize(new Normalize({ 0.485, 0.456, 0.406}, { 0.229, 0.224, 0.225 }));
    std::shared_ptr<TensorTransform> normalize1(new Normalize({ 0, 0, 0}, { 255, 255, 255 }));
    std::shared_ptr<TensorTransform> hwc2chw(new HWC2CHW());
    // define preprocessor
    ds::Execute preprocessor({decode, resize, normalize1, normalize, hwc2chw});
    auto data_set = FLAGS_image_path;
    std::map<double, double> costTime_map;
    std::vector<std::string> dirs;
    if (FLAGS_dataset_name == "TUM") {
        data_set = FLAGS_image_path + "/TUM/rgbd_dataset_freiburg2_desk_with_person";
        dirs.emplace_back("TUM");
    } else if (FLAGS_dataset_name == "Kitti") {
        data_set = FLAGS_image_path + "/Kitti_raw_data/";
        dirs = GetAlldir(data_set, FLAGS_dataset_name);
    } else if (FLAGS_dataset_name == "Sintel") {
        data_set = FLAGS_image_path + "/Sintel/final_left/";
        dirs = GetAlldir(data_set, FLAGS_dataset_name);
    } else {
        std::cout << "dataset error" << std::endl;
        return -1;
    }
    for (const auto &dir : dirs) {
        std::vector<std::string> images;
        if (FLAGS_dataset_name == "TUM") images = GetAlldir(data_set, "TUM");
        else
        images = GetAllFiles(data_set + dir);
        for (const auto& image_file : images) {
            struct timeval start;
            struct timeval end;
            double startTime_ms;
            double endTime_ms;
            // prepare input
            std::vector<ms::MSTensor> outputs;
            std::vector<ms::MSTensor> inputs;
            std::cout << "Start predict input files:" << image_file << std::endl;
            // read image file and preprocess
            auto image = ReadFileToTensor(image_file);
            ret = preprocessor(image, &image);
            if (ret != ms::kSuccess) {
                std::cout << "Image preprocess failed." << std::endl;
                return 1;
            }
            inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                image.Data().get(), image.DataSize());
            gettimeofday(&start, NULL);
            ret = midas.Predict(inputs, &outputs);
            gettimeofday(&end, NULL);
            if (ret != ms::kSuccess) {
                std::cout << "Predict model failed." << std::endl;
                return 1;
            }
            startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
            endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
            costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
            WriteResult(image_file, outputs, FLAGS_dataset_name, dir);
        }
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
    timeCost << "NN inference cost average time: " << average << " ms of infer_count " << inferCount << std::endl;
    std::cout << "NN inference cost average time: " << average << "ms of infer_count " << inferCount << std::endl;
    std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
    fileStream << timeCost.str();
    fileStream.close();
    costTime_map.clear();
    return 0;
}

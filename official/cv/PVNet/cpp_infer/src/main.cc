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
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "common_inc/infer.h"

DEFINE_string(model_path, "", "model path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(aipp_path, "./aipp.cfg", "aipp path");
DEFINE_string(device_type, "CPU", "device type");

int main(int argc, char **argv) {
    if (!ParseCommandLineFlags(argc, argv)) {
        std::cout << "Failed to parse args" << std::endl;
        return 1;
    }
    if (RealPath(FLAGS_model_path).empty()) {
      std::cout << "Invalid model" << std::endl;
      return 1;
    }

    Model model;
    Status ret;
    if (!LoadModel(FLAGS_model_path, FLAGS_device_type, FLAGS_device_id, &model)) {
        std::cout << "Failed to load model " << FLAGS_model_path << ", device id: " << FLAGS_device_id
                << ", device type: " << FLAGS_device_type;
        return 1;
    }

    std::vector<MSTensor> modelInputs = model.GetInputs();

    auto all_files = GetAllFiles(FLAGS_dataset_path);
    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }
    auto decode = Decode();
    auto rescale = Rescale(1.0 / 255.0, 0);
    auto hwc2chw = HWC2CHW();
    auto normalize = Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});

    mindspore::dataset::Execute transformDecode(decode);
    mindspore::dataset::Execute transform({rescale, normalize, hwc2chw});

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

        transformDecode(image, &image);
        std::vector<int64_t> shape = image.Shape();
        transform(image, &image);

        inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(),
                            image.Data().get(), image.DataSize());

        gettimeofday(&start, NULL);
        model.Predict(inputs, &outputs);
        std::cout << " infer result:" << all_files[i] << std::endl;
        gettimeofday(&end, NULL);

        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
        WriteResult(all_files[i], outputs);
    }
    double average = 0.0;
    double Fps = 0.0;
    int infer_cnt = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        std::cout << "time_time_1"<< iter->second << "time_time_0 "
                  << iter->first << iter->second - iter->first << std::endl;
        diff = iter->second - iter->first;
        average += diff;
        infer_cnt++;
    }
    Fps = infer_cnt*1000 / average;
    average = average / infer_cnt;

    std::stringstream timeCost;
    std::stringstream fps;
    fps << "Image processing speed is: " << Fps << "imgs/s" << std::endl;
    timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << infer_cnt << std::endl;
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << infer_cnt << std::endl;
    std::string file_name = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
    file_stream << fps.str();
    file_stream << timeCost.str();
    file_stream.close();
    costTime_map.clear();
    return 0;
}

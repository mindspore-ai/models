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

DEFINE_string(model1_path, "", "coarse model path");
DEFINE_string(model2_path, "", "finet model path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(device_type, "CPU", "device type");

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }
  if (RealPath(FLAGS_model1_path).empty() || RealPath(FLAGS_model2_path).empty()) {
    std::cout << "Invalid model" << std::endl;
    return 1;
  }

  Model model1;
  if (!LoadModel(FLAGS_model1_path, FLAGS_device_type, FLAGS_device_id, &model1)) {
    std::cout << "Failed to load model1 " << FLAGS_model1_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }
  Model model2;
  if (!LoadModel(FLAGS_model2_path, FLAGS_device_type, FLAGS_device_id, &model2)) {
    std::cout << "Failed to load model2 " << FLAGS_model2_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }

  Status ret;
  std::vector<MSTensor> model1Inputs = model1.GetInputs();
  std::vector<MSTensor> model2Inputs = model2.GetInputs();

  auto all_files = GetAllFiles(FLAGS_dataset_path);
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
    std::vector<MSTensor> inputs1;
    std::vector<MSTensor> inputs2;
    std::vector<MSTensor> coarseOutputs;
    std::vector<MSTensor> outputs;

    std::cout << "Start predict input files:" << all_files[i] << std::endl;
    mindspore::MSTensor image = ReadFileToTensor(all_files[i]);
    gettimeofday(&start, NULL);
    inputs1.emplace_back(model1Inputs[0].Name(), model1Inputs[0].DataType(), model1Inputs[0].Shape(),
                         image.Data().get(), image.DataSize());
    model1.Predict(inputs1, &coarseOutputs);

    inputs2.emplace_back(model2Inputs[0].Name(), model2Inputs[0].DataType(), model2Inputs[0].Shape(),
                         image.Data().get(), image.DataSize());
    inputs2.emplace_back(model2Inputs[1].Name(), model2Inputs[1].DataType(), model2Inputs[1].Shape(),
                         coarseOutputs[0].Data().get(), coarseOutputs[0].DataSize());
    model2.Predict(inputs2, &outputs);

    gettimeofday(&end, NULL);

    startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;  // 1000000, 1000 to convert time to ms
    endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;        // 1000000, 1000 to convert time to ms
    costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
    WriteResult(all_files[i], outputs);
  }
  double average = 0.0;
  int infer_cnt = 0;
  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = iter->second - iter->first;
    average += diff;
    infer_cnt++;
  }

  average = average / infer_cnt;
  std::stringstream timeCost;
  timeCost << "NN inference cost average time: " << average << " ms of infer_count " << infer_cnt << std::endl;
  std::cout << "NN inference cost average time: " << average << "ms of infer_count " << infer_cnt << std::endl;

  std::string file_name = "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
  file_stream << timeCost.str();
  file_stream.close();
  costTime_map.clear();
  return 0;
}

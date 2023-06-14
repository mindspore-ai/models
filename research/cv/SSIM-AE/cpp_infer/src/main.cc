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

#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "common_inc/infer.h"

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(device_type, "CPU", "device type");
DEFINE_string(need_preprocess, "n", "need preprocess or not");

int main(int argc, char **argv) {
  std::cout << "Start c+" << std::endl;
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  mindspore::Model model;
  if (!LoadModel(FLAGS_mindir_path, FLAGS_device_type, FLAGS_device_id, &model)) {
    std::cout << "Failed to load model " << FLAGS_mindir_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }
  Status ret;

  std::vector<MSTensor> model_inputs = model.GetInputs();
  auto all_files = GetAllFiles(FLAGS_dataset_path);
  if (all_files.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
  }

  std::map<double, double> costTime_map;
  size_t size = all_files.size();

  auto decode = Decode();
  auto swapredblue = SwapRedBlue();
  auto resize = Resize({256, 256});
  auto normalize = Normalize({127.5, 127.5, 127.5}, {127.5, 127.5, 127.5});
  auto hwc2chw = HWC2CHW();
  Execute preprocess({decode, swapredblue, resize, normalize, hwc2chw});

  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTime_ms;
    double endTime_ms;
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    std::cout << "Start predict input files:" << all_files[i] << std::endl;

    auto img = MSTensor();
    if (FLAGS_need_preprocess == "y") {
      ret = preprocess(ReadFileToTensor(all_files[i]), &img);
      if (ret != kSuccess) {
        std::cout << "preprocess " << all_files[i] << " failed." << std::endl;
        return 1;
      }
    } else {
      img = ReadFileToTensor(all_files[i]);
    }

    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(), img.Data().get(),
                        img.DataSize());
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
    WriteResult(all_files[i], outputs, "./postprocess_result");
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
  std::string file_name = std::string("test_perform_static.txt");
  std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
  file_stream << timeCost.str();
  file_stream.close();
  costTime_map.clear();
  return 0;
}

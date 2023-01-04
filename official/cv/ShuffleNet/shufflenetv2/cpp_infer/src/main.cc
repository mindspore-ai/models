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

DEFINE_string(mindir_path, "", "model path");
DEFINE_string(dataset_name, "", "dataset name");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(device_type, "CPU", "device type");

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid model" << std::endl;
    return 1;
  }

  auto ascend310_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310_info->SetDeviceID(FLAGS_device_id);
  ascend310_info->SetOpSelectImplMode("high_precision");
  mindspore::Model model;
  if (!LoadModel(FLAGS_mindir_path, FLAGS_device_type, FLAGS_device_id, ascend310_info, &model)) {
    std::cout << "Failed to load model " << FLAGS_mindir_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }
  Status ret;

  auto all_files = GetAllFiles(FLAGS_dataset_path);
  if (all_files.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
  }

  std::vector<MSTensor> modelInputs = model.GetInputs();

  std::shared_ptr<TensorTransform> decode = std::make_shared<Decode>();
  std::shared_ptr<TensorTransform> resize = std::make_shared<Resize>(std::vector<int>{256});
  std::shared_ptr<TensorTransform> centercrop = std::make_shared<CenterCrop>(std::vector<int>{224});

  std::shared_ptr<TensorTransform> normalize =
    std::make_shared<Normalize>(std::vector<float>{123.675, 116.28, 103.53}, std::vector<float>{58.395, 57.12, 57.375});

  std::shared_ptr<TensorTransform> hwc2chw = std::make_shared<HWC2CHW>();

  std::vector<std::shared_ptr<TensorTransform>> trans_list;
  trans_list = {decode, resize, centercrop, normalize, hwc2chw};

  mindspore::dataset::Execute SingleOp(trans_list);

  std::map<double, double> costTime_map;

  size_t size = all_files.size();
  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};

    double startTime_ms;
    double endTime_ms;

    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;

    std::cout << "Start predict input files:" << all_files[i] << std::endl;

    mindspore::MSTensor image = ReadFileToTensor(all_files[i]);
    SingleOp(image, &image);

    inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(), image.Data().get(),
                        image.DataSize());

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
    average += iter->second - iter->first;
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

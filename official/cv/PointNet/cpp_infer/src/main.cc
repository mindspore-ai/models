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
DEFINE_string(aipp_path, "", "aipp path");
DEFINE_string(cpu_dvpp, "", "cpu or dvpp process");
DEFINE_int32(image_height, 32, "image height");
DEFINE_int32(image_width, 32, "image width");
DEFINE_string(device_type, "CPU", "device type");

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  ascend310->SetBufferOptimizeMode("off_optimize");
  if (FLAGS_cpu_dvpp == "DVPP") {
    if (RealPath(FLAGS_aipp_path).empty()) {
      std::cout << "Invalid aipp path" << std::endl;
      return 1;
    } else {
      ascend310->SetInsertOpConfigPath(FLAGS_aipp_path);
    }
  }

  Status ret;
  Model model;
  if (!LoadModel(FLAGS_mindir_path, FLAGS_device_type, FLAGS_device_id, ascend310, &model)) {
    std::cout << "Failed to load model " << FLAGS_mindir_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }

  auto all_files = GetAllFiles(FLAGS_dataset_path);
  std::map<double, double> costTime_map;
  size_t size = all_files.size();

  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    std::cout << "Start predict input files:" << all_files[i] << std::endl;
    if (FLAGS_cpu_dvpp == "DVPP") {
      std::shared_ptr<TensorTransform> decode(new Decode());
      auto resizeShape = {FLAGS_image_height, FLAGS_image_width};
      std::shared_ptr<TensorTransform> resize(new Resize(resizeShape));
      // Execute composeDecode({decode, resize});
      Execute composeDecode({});
      auto imgDvpp = std::make_shared<MSTensor>();
      inputs.emplace_back(imgDvpp->Name(), imgDvpp->DataType(), imgDvpp->Shape(),
                        imgDvpp->Data().get(), imgDvpp->DataSize());
    } else if (FLAGS_cpu_dvpp == "CPU")  {
      std::shared_ptr<TensorTransform> decode(new Decode());
      std::shared_ptr<TensorTransform> hwc2chw(new HWC2CHW());
      std::shared_ptr<TensorTransform> normalize(
      new Normalize({123.675, 116.28, 103.53}, {58.395, 57.120, 57.375}));
      auto resizeShape = {FLAGS_image_height, FLAGS_image_width};
      std::shared_ptr<TensorTransform> resize(new Resize(resizeShape));
      auto resizeShape1 = {1, FLAGS_image_height};
      std::shared_ptr<TensorTransform> reshape_one_channel(new Resize(resizeShape1));
      Execute composeDecode({decode, resize, normalize, hwc2chw, reshape_one_channel});
      auto img = MSTensor();
      auto image = ReadFileToTensor(all_files[i]);
      composeDecode(image, &img);
      std::vector<MSTensor> model_inputs = model.GetInputs();
      inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                       img.Data().get(), img.DataSize());
    } else  {
      auto image = ReadFileToTensor(all_files[i]);
      std::vector<MSTensor> model_inputs = model.GetInputs();
      inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                          image.Data().get(), image.DataSize());
    }

    gettimeofday(&start, nullptr);
    ret = model.Predict(inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Predict " << all_files[i] << " failed." << std::endl;
      return 1;
    }
    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
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
  std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << std::endl;
  std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
  fileStream << timeCost.str();
  fileStream.close();
  costTime_map.clear();
  return 0;
}

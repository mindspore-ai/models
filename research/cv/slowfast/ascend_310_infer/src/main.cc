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
using mindspore::MSTensor;
using mindspore::dataset::Execute;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(input0_path, ".", "input0 path");
DEFINE_int32(device_id, 0, "device id");

bool is_file_exist(const char *fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}

std::string base_name(std::string const & path) {
  return path.substr(path.find_last_of("/\\") + 1);
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  context->MutableDeviceInfo().push_back(ascend310);
  mindspore::Graph graph;
  Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);

  Model model;
  Status ret = model.Build(GraphCell(graph), context);
  if (ret != kSuccess) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }

  std::vector<MSTensor> model_inputs = model.GetInputs();
  if (model_inputs.empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }

  auto input0_files = GetAllFiles(FLAGS_input0_path);

  if (input0_files.empty()) {
    std::cout << "ERROR: input data empty." << std::endl;
    return 1;
  }
  std::map<double, double> costTime_map;
  size_t singleParamCount = input0_files.size() / 7 - 1;
  for (size_t i = 0; i < singleParamCount; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    // slowpath
    auto slowpathFile = FLAGS_input0_path + "/slowpath/ava_val_" + std::to_string(i) + ".bin";
    auto fastpathFile = FLAGS_input0_path + "/fastpath/ava_val_" + std::to_string(i) + ".bin";
    auto boxesFile = FLAGS_input0_path + "/boxes/ava_val_" + std::to_string(i) + ".bin";
    auto slowpath = ReadFileToTensor(slowpathFile);
    auto fastpath = ReadFileToTensor(fastpathFile);
    // boxes
    auto boxes = ReadFileToTensor(boxesFile);
    // Log begin.
    std::cout << "Start predict input files:" << slowpathFile << std::endl;
    std::cout << "Start predict input files:" << fastpathFile << std::endl;
    std::cout << "Start predict input files:" << boxesFile << std::endl;
    // Set input.
    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                        slowpath.Data().get(), slowpath.DataSize());
    inputs.emplace_back(model_inputs[1].Name(), model_inputs[1].DataType(), model_inputs[1].Shape(),
                        fastpath.Data().get(), fastpath.DataSize());
    inputs.emplace_back(model_inputs[2].Name(), model_inputs[2].DataType(), model_inputs[2].Shape(),
                        boxes.Data().get(), boxes.DataSize());
    gettimeofday(&start, nullptr);
    ret = model.Predict(inputs, &outputs);
    std::cout << "Finish single predict" << std::endl;
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Predict " << slowpathFile << std::endl;
      std::cout << "Predict " << fastpathFile << std::endl;
      std::cout << "Predict " << boxesFile << std::endl;
      return 1;
    }
    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
    WriteResult(slowpathFile, outputs);
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

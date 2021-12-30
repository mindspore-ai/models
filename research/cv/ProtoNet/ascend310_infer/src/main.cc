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

#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/dataset/execute.h"
#include "include/api/types.h"
#include "include/dataset/vision_ascend.h"
#include "include/dataset/vision.h"
#include "../inc/utils.h"

namespace ms = mindspore;
namespace ds = mindspore::dataset;


DEFINE_string(mindir_path, ".", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");




int main(int argc, char **argv) {
  using std::cout;
  using std::endl;
  using std::string;
  using std::vector;
  using std::make_shared;
  using std::ofstream;
  using std::stringstream;
  using std::map;
  using std::pair;
  using std::ios;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  cout << FLAGS_mindir_path << endl;
  cout << FLAGS_dataset_path << endl;
  cout << FLAGS_device_id << endl;
  // set context
  auto context = make_shared<ms::Context>();
  auto ascend310_info = make_shared<ms::Ascend310DeviceInfo>();
  ascend310_info->SetDeviceID(FLAGS_device_id);
  context->MutableDeviceInfo().push_back(ascend310_info);
  // define model
  ms::Graph graph;
  ms::Status ret = ms::Serialization::Load(FLAGS_mindir_path, ms::ModelType::kMindIR, &graph);
  if (ret != ms::kSuccess) {
    cout << "Load model failed." << endl;
    return 1;
  }
  ms::Model protoNet;
  // build model
  ret = protoNet.Build(ms::GraphCell(graph), context);
  if (ret != ms::kSuccess) {
    cout << "Build model failed." << endl;
    return 1;
  }
  // get model input info
  vector<ms::MSTensor> model_inputs = protoNet.GetInputs();
  if (model_inputs.empty()) {
    cout << "Invalid model, inputs is empty." << endl;
    return 1;
  }
  // load data
  vector<string> images = GetAllFiles(FLAGS_dataset_path);
  map<double, double> costTime_map;
  size_t size = images.size();
  // start infer
  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    vector<ms::MSTensor> outputs;
    vector<ms::MSTensor> inputs;
    auto image = ReadFile(images[i]);
    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                        image.Data().get(), image.DataSize());
    gettimeofday(&start, nullptr);
    ret = protoNet.Predict(inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != ms::kSuccess) {
      cout << "Predict model failed." << endl;
      return 1;
    }
    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(pair<double, double>(startTimeMs, endTimeMs));
    WriteResult(images[i], outputs);
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
  stringstream timeCost;
  timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << inferCount << endl;
  cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << endl;
  string fileName = "./time_Result" + string("/test_perform_static.txt");
  ofstream fileStream(fileName.c_str(), ios::trunc);
  fileStream << timeCost.str();
  fileStream.close();
  costTime_map.clear();
  return 0;
}



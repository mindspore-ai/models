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

#include <dirent.h>
#include <gflags/gflags.h>
#include <sys/time.h>
#include <fstream>
#include <algorithm>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "inc/utils.h"
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/types.h"

using mindspore::Context;
using mindspore::DataType;
using mindspore::GraphCell;
using mindspore::Model;
using mindspore::ModelType;
using mindspore::MSTensor;
using mindspore::Serialization;
using mindspore::Status;

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(batch_mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_string(image_path, ".", "image path");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }
  if (RealPath(FLAGS_batch_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  context->MutableDeviceInfo().push_back(ascend310);

  std::vector<mindspore::Graph> graph;
  Serialization::Load({FLAGS_mindir_path, FLAGS_batch_mindir_path}, ModelType::kMindIR, &graph);

  Model model;
  Status ret = model.Build(GraphCell(graph[0]), context);
  if (ret.IsError()) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }
  if (!model.HasPreprocess()) {
    std::cout << "data preprocess not exists in MindIR " << std::endl;
    return 1;
  }

  Model model2;
  ret = model2.Build(GraphCell(graph[1]), context);
  if (ret.IsError()) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }
  if (!model2.HasPreprocess()) {
    std::cout << "data preprocess not exists in MindIR " << std::endl;
    return 1;
  }

  // preprocess and predict with batch 1
  std::vector<std::vector<MSTensor>> inputs1;
  MSTensor *t1 = MSTensor::CreateTensorFromFile(FLAGS_image_path);
  inputs1 = {{*t1}};
  std::vector<MSTensor> outputs1;

  ret = model.Preprocess(inputs1, &outputs1);
  if (ret.IsError()) {
    std::cout << ret.GetErrDescription() << std::endl;
    std::cout << "ERROR: Preprocess failed." << std::endl;
    return 1;
  }

  std::vector<MSTensor> outputs1_1;
  ret = model.Predict(outputs1, &outputs1_1);
  if (ret.IsError()) {
    std::cout << ret.GetErrDescription() << std::endl;
    std::cout << "ERROR: Predict failed." << std::endl;
    return 1;
  }

  std::ofstream o1("result1.txt", std::ios::out);
  o1.write(reinterpret_cast<const char *>(outputs1_1[0].MutableData()),
           std::streamsize(outputs1_1[0].DataSize()));

  // check shape
  auto shape1 = outputs1_1[0].Shape();
  std::cout << "outputs1_1 shape: " << std::endl;
  for (auto s : shape1) {
    std::cout << s << ", ";
  }
  std::cout << std::endl;
  MSTensor::DestroyTensorPtr(t1);

  // preprocess and predict with batch 3
  std::vector<std::vector<MSTensor>> inputs2;
  MSTensor *t2 = MSTensor::CreateTensorFromFile(FLAGS_image_path);
  MSTensor *t3 = MSTensor::CreateTensorFromFile(FLAGS_image_path);
  MSTensor *t4 = MSTensor::CreateTensorFromFile(FLAGS_image_path);
  inputs2 = {{*t2}, {*t3}, {*t4}};

  std::vector<MSTensor> outputs2;
  ret = model2.PredictWithPreprocess(inputs2, &outputs2);
  if (ret.IsError()) {
    std::cout << ret.GetErrDescription() << std::endl;
    std::cout << "ERROR: Predict failed." << std::endl;
    return 1;
  }
  std::ofstream o2("result2.txt", std::ios::out);
  o2.write(reinterpret_cast<const char *>(outputs2[0].MutableData()),
           std::streamsize(outputs2[0].DataSize()));

  // check shape
  auto shape2 = outputs2[0].Shape();
  std::cout << "outputs2 shape: " << std::endl;
  for (auto s : shape2) {
    std::cout << s << ", ";
  }
  std::cout << std::endl;
  MSTensor::DestroyTensorPtr(t2);
  MSTensor::DestroyTensorPtr(t3);
  MSTensor::DestroyTensorPtr(t4);

  return 0;
}

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
#include <math.h>
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
#include "include/dataset/transforms.h"
#include "include/dataset/vision_ascend.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"
#include "inc/utils.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::DataType;
using mindspore::dataset::Execute;
using mindspore::dataset::InterpolationMode;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::Decode;


DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_int32(image_height, 640, "image height");
DEFINE_int32(image_width, 640, "image width");


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  ascend310->SetPrecisionMode("allow_fp32_to_fp16");
  ascend310->SetOpSelectImplMode("high_precision");
  ascend310->SetBufferOptimizeMode("off_optimize");
  context->MutableDeviceInfo().push_back(ascend310);
  mindspore::Graph graph;
  Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);
  Model model;
  Status ret = model.Build(GraphCell(graph), context);
  if (ret != kSuccess) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }

  auto all_files = GetAllFiles(FLAGS_dataset_path);
  std::map<double, double> costTime_map;
  size_t size = all_files.size();
  std::shared_ptr<TensorTransform> decode(new Decode());
  auto resize = Resize({FLAGS_image_height, FLAGS_image_width});
  Execute composeDecode({decode});

  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    auto imgDecode = MSTensor();
    auto img = MSTensor();
    composeDecode(ReadFileToTensor(all_files[i]), &imgDecode);
    std::vector<int64_t> shape = imgDecode.Shape();

    if ((static_cast<int> (shape[0]) < static_cast<int>(FLAGS_image_height)) &&
        (static_cast<int> (shape[1]) < static_cast<int>(FLAGS_image_width))) {
      resize = Resize({FLAGS_image_height, FLAGS_image_width}, InterpolationMode::kCubic);
    } else if ((static_cast<int> (shape[0]) > static_cast<int>(FLAGS_image_height)) &&
               (static_cast<int> (shape[1]) > static_cast<int>(FLAGS_image_width))) {
      resize = Resize({FLAGS_image_height, FLAGS_image_width}, InterpolationMode::kNearestNeighbour);
    } else {
      resize = Resize({FLAGS_image_height, FLAGS_image_width}, InterpolationMode::kLinear);
    }
    if ((sizeof(shape)/sizeof(shape[0])) <= 2) {
      std::cout << "image channels is not 3." << std::endl;
      return 1;
    }
    Execute transform({resize});
    transform(imgDecode, &img);

    size_t buffer_size = img.DataSize();
    std::vector<int64_t> img_shape = img.Shape();
    std::vector<MSTensor> model_inputs = model.GetInputs();
    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                        img.Data().get(), img.DataSize());
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
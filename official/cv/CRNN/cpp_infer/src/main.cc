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

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(device_type, "CPU", "device type");
DEFINE_string(precision_mode, "preferred_fp32", "precision mode");
DEFINE_string(op_select_impl_mode, "", "op select impl mode");
DEFINE_string(aipp_path, "", "aipp config file");

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }
  auto ascend = std::make_shared<mindspore::AscendDeviceInfo>();
  ascend->SetDeviceID(FLAGS_device_id);
  ascend->SetInsertOpConfigPath({FLAGS_aipp_path});
  ascend->SetPrecisionMode("preferred_fp32");
  ascend->SetOpSelectImplMode("high_precision");

  Model model;
  if (!LoadModel(FLAGS_mindir_path, FLAGS_device_type, FLAGS_device_id, ascend, &model)) {
    std::cout << "Failed to load model " << FLAGS_mindir_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }
  Status ret;

  std::vector<MSTensor> modelInputs = model.GetInputs();

  auto allFiles = GetAllFiles(FLAGS_dataset_path);
  if (allFiles.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
  }

  Execute compose({std::shared_ptr<Decode>(new Decode()), std::shared_ptr<SwapRedBlue>(new SwapRedBlue()),
                   std::shared_ptr<Resize>(new Resize({32, 100})),
                   std::shared_ptr<Normalize>(new Normalize({127.5, 127.5, 127.5}, {127.5, 127.5, 127.5})),
                   std::shared_ptr<HWC2CHW>(new HWC2CHW())});
  Execute composeCast(std::shared_ptr<TypeCast>(new TypeCast(DataType::kNumberTypeFloat16)));

  struct timeval start;
  struct timeval end;
  double startTime_ms;
  double endTime_ms;
  std::map<double, double> costTime_map;
  size_t size = allFiles.size();

  for (size_t i = 0; i < size; ++i) {
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    std::cout << "Start predict input files:" << allFiles[i] << std::endl;
    std::string suffix = allFiles[i].substr(allFiles[i].rfind("."));
    if (suffix != ".jpg" && suffix != ".png" && suffix != ".JPG" && suffix != ".PNG") {
      std::cout << "wrong file format: " << allFiles[i] << std::endl;
      continue;
    }

    mindspore::MSTensor img;
    compose(ReadFileToTensor(allFiles[i]), &img);

    inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(), img.Data().get(),
                        img.DataSize());

    gettimeofday(&start, NULL);
    ret = model.Predict(inputs, &outputs);
    gettimeofday(&end, NULL);
    if (ret != kSuccess) {
      std::cout << "Predict " << allFiles[i] << " failed." << std::endl;
      return 1;
    }
    startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
    int rst = WriteResult(allFiles[i], outputs);
    if (rst != 0) {
      std::cout << "write result failed." << std::endl;
      return rst;
    }
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

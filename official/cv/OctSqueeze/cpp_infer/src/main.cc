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
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <iostream>
#include <iosfwd>
#include <sstream>
#include <string>
#include <vector>

#include "../include/utils.h"
#include "common_inc/infer.h"

DEFINE_string(model_path, "./pretrained_ckpts/octsqueeze.mindir", "model path");
DEFINE_string(datasets_dir, "./datasets/test", "dataset dir");
DEFINE_int32(batch_size, 98304, "batch size");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(device_target, "Ascend310", "device target");
DEFINE_string(device_type, "CPU", "device type");

void ComputeTimeslot(const struct timeval &bgn, const struct timeval &end, std::map<double, double> &timeslots) {
  double bgn_ms = (bgn.tv_sec * 1e+6 + bgn.tv_usec) / 1e+3;  // ms
  double end_ms = (end.tv_sec * 1e+6 + end.tv_usec) / 1e+3;  // ms
  auto timeslot = std::pair<double, double>(bgn_ms, end_ms);
  timeslots.insert(timeslot);
}

double AverageTimeslots(const std::map<double, double> &timeslots) {
  double sum = 0;
  int num = timeslots.size();
  for (auto iter = timeslots.begin(); iter != timeslots.end(); ++iter) {
    double dif = iter->second - iter->first;
    sum += dif;
  }
  return (sum / num);
}

void SerializeResult(const std::vector<float> &bitrateBpips, const std::vector<double> &bitrateTimelot,
                     std::vector<std::string> &precisionList) {
  std::stringstream io_stream;
  for (size_t i = 0; i < precisionList.size(); i++) {
    std::cout << "At precision " << precisionList[i] << ": ";
    std::cout << "bpip =  " << bitrateBpips[i] << "; ";
    std::cout << "each frame cost " << bitrateTimelot[i] << " ms" << std::endl;
    io_stream << "At precision " << precisionList[i] << ": ";
    io_stream << "bpip =  " << bitrateBpips[i] << "; ";
    io_stream << "each frame cost " << bitrateTimelot[i] << " ms" << std::endl;
  }

  std::string file_name = "./logs/test_performance.txt";
  std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
  file_stream << io_stream.str();
  file_stream.close();
}

void check_cwd() {
  char cwd[PATH_MAX];
  getcwd(cwd, sizeof(cwd));
  std::cout << "cwd: " << cwd << std::endl;
}

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }

  check_cwd();

  auto path = RealPath(FLAGS_model_path);
  if (path.empty()) {
    std::cout << "Invalid model" << std::endl;
    return 1;
  }
  auto ascend = std::make_shared<mindspore::AscendDeviceInfo>();
  ascend->SetDeviceID(FLAGS_device_id);
  ascend->SetPrecisionMode("preferred_fp32");
  ascend->SetOpSelectImplMode("high_precision");
  Model model;
  if (!LoadModel(FLAGS_model_path, FLAGS_device_type, FLAGS_device_id, ascend, &model)) {
    std::cout << "Failed to load model " << FLAGS_model_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }
  Status ret;

  std::vector<std::string> precisionList = {"0.01", "0.02", "0.04", "0.08"};
  std::vector<float> bitrateBpips;
  std::vector<double> bitrateTimelot;

  for (auto precision_iter = precisionList.begin(); precision_iter != precisionList.end(); precision_iter++) {
    std::string precision = precision_iter->data();
    std::cout << "Start processing precision: " << precision << std::endl;

    auto data_dirs = GetDataDirs(FLAGS_datasets_dir, precision);
    if (data_dirs.empty()) {
      std::cout << "ERROR: no input data." << std::endl;
      return 1;
    }

    std::map<double, double> timeslot_map;
    std::vector<float> bpips;
    for (size_t i = 0; i < data_dirs.size(); ++i) {
      std::cout << "Start predict data_dir=" << data_dirs[i] << std::endl;

      // Read raw data
      int32_t points_num = 0;
      std::vector<float> input;
      std::vector<uint32_t> gt;
      GetDataSample(data_dirs[i], points_num, input, gt);

      size_t batch_num = ceil(input.size() / 24.0 / FLAGS_batch_size);

      std::shared_ptr<float> sptrInput(new float[batch_num * 24 * FLAGS_batch_size], std::default_delete<float[]>());
      std::shared_ptr<float> sptrGt(new float[gt.size()], std::default_delete<float[]>());

      for (size_t j = 0; j < input.size(); ++j) {
        sptrInput.get()[j] = input[j];
      }

      for (size_t j = 0; j < gt.size(); ++j) {
        sptrGt.get()[j] = gt[j];
      }

      struct timeval bgn_tv, end_tv;
      std::vector<MSTensor> inputs, outputs;

      std::vector<MSTensor> modelInputs = model.GetInputs();

      gettimeofday(&bgn_tv, NULL);

      size_t batch_len = FLAGS_batch_size * 24 * sizeof(float);
      std::vector<MSTensor> batch_output;
      for (size_t batch_idx = 0; batch_idx < batch_num; ++batch_idx) {
        const void *batch_ptr = (const void *)(sptrInput.get() + batch_idx * FLAGS_batch_size * 24);
        MSTensor msTensor(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(), batch_ptr,
                          batch_len);
        inputs.emplace_back(msTensor);

        model.Predict(inputs, &batch_output);

        outputs.insert(outputs.end(), batch_output.begin(), batch_output.end());
        inputs.clear();
      }

      gettimeofday(&end_tv, NULL);
      ComputeTimeslot(bgn_tv, end_tv, timeslot_map);
      float bpp;
      bpp = WriteResult(data_dirs[i], outputs, gt, points_num);
      bpips.push_back(bpp);
    }

    float bpipsSum = 0;
    bpipsSum = std::accumulate(bpips.begin(), bpips.end(), bpipsSum);
    float bpipMean = bpipsSum / data_dirs.size();
    std::cout << "At precision " << precision << " average bpip: " << bpipMean << "; ";
    bitrateBpips.push_back(bpipMean);

    double timeslot = AverageTimeslots(timeslot_map);
    int sample_num = timeslot_map.size();
    std::cout << "NN inference cost average time: " << timeslot << " ms of sample_num " << sample_num << std::endl;
    bitrateTimelot.push_back(timeslot);
    timeslot_map.clear();
  }
  SerializeResult(bitrateBpips, bitrateTimelot, precisionList);

  return 0;
}

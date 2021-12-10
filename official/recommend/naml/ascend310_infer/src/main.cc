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
#include <string.h>
#include <iostream>
#include <algorithm>

#include <string>
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

//*****     bs=16     *****//
DEFINE_string(news_mindir, "./", "om model path.");
DEFINE_string(user_mindir, "./", "om model path.");
DEFINE_int32(batch_size, 16, "batch size");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(base_path, "./", "dataset base path.");

std::vector<std::string> news_input0_0_files;
auto context = std::make_shared<Context>();

struct timeval total_start;
struct timeval total_end;
std::map<double, double> model0_cost_time;
std::map<double, double> model1_cost_time;

int InitModel(const std::string &model_path, Model *model) {
  mindspore::Graph graph;
  Serialization::Load(model_path, ModelType::kMindIR, &graph);
  Status ret = model->Build(GraphCell(graph), context);
  if (ret != kSuccess) {
    std::cout << "ERROR: Build model failed." << std::endl;
    return 1;
  }
  return 0;
}

void GetAveTime(const std::map<double, double> &costTime_map, int *inferCount, double *average) {
  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = 0.0;
    diff = iter->second - iter->first;
    (*average) += diff;
    (*inferCount)++;
  }
  (*average) /= (*inferCount);
  return;
}

void InitInputs(size_t input_index, const std::vector<MSTensor> &model0_inputs, std::vector<MSTensor> *inputs) {
  std::string news_dataset_path = FLAGS_base_path + "/news_test_data";
  auto file0_name = news_dataset_path + "/00_category_data/naml_news_" + std::to_string(input_index) + ".bin";
  auto file1_name = news_dataset_path + "/01_subcategory_data/naml_news_" + std::to_string(input_index) + ".bin";
  auto file2_name = news_dataset_path + "/02_title_data/naml_news_" + std::to_string(input_index) + ".bin";
  auto file3_name = news_dataset_path + "/03_abstract_data/naml_news_" + std::to_string(input_index) + ".bin";

  auto news_input0 = ReadFileToTensor(file0_name);
  auto news_input1 = ReadFileToTensor(file1_name);
  auto news_input2 = ReadFileToTensor(file2_name);
  auto news_input3 = ReadFileToTensor(file3_name);

  inputs->emplace_back(model0_inputs[0].Name(), model0_inputs[0].DataType(), model0_inputs[0].Shape(),
                      news_input0.Data().get(), news_input0.DataSize());
  inputs->emplace_back(model0_inputs[1].Name(), model0_inputs[1].DataType(), model0_inputs[1].Shape(),
                      news_input1.Data().get(), news_input1.DataSize());
  inputs->emplace_back(model0_inputs[2].Name(), model0_inputs[2].DataType(), model0_inputs[2].Shape(),
                      news_input2.Data().get(), news_input2.DataSize());
  inputs->emplace_back(model0_inputs[3].Name(), model0_inputs[3].DataType(), model0_inputs[3].Shape(),
                      news_input3.Data().get(), news_input3.DataSize());
  return;
}

void InitNewsDict(const std::vector<std::string> &news_input0_0_files, const std::vector<MSTensor> &model0_inputs,
                  std::map<uint32_t, MSTensor> *news_dict, Model *model0) {
  int count = 0;
  for (size_t i = 0; i < news_input0_0_files.size() - 1; ++i) {
    struct timeval start;
    struct timeval end;
    double startTime_ms;
    double endTime_ms;
    std::vector <MSTensor> inputs0;
    std::vector <MSTensor> outputs0;

    // init inputs by model0 input for each iter
    InitInputs(i, model0_inputs, &inputs0);

    // get model0 outputs
    gettimeofday(&total_start, nullptr);
    gettimeofday(&start, nullptr);
    Status ret0 = model0->Predict(inputs0, &outputs0);
    gettimeofday(&end, nullptr);
    if (ret0 != kSuccess) {
      std::cout << "ERROR: Predict model0 failed." << std::endl;
      return;
    }
    // init news_dict
    auto file_name = FLAGS_base_path + "/news_id_data/naml_news_" + std::to_string(count++) + ".bin";
    auto nid = ReadFileToTensor(file_name);
    auto nid_addr = reinterpret_cast<uint32_t *>(nid.MutableData());
    // output0 size is 1, shape: batch_size * 400
    auto outputs0_addr = reinterpret_cast<float *>(outputs0[0].MutableData());

    for (int k = 0; k < FLAGS_batch_size; ++k) {
      MSTensor ret("", mindspore::DataType::kNumberTypeFloat32, {400}, nullptr, sizeof(float)* 400);
      uint32_t *addr = reinterpret_cast<uint32_t *>(ret.MutableData());
      memcpy(addr, outputs0_addr, 400 * sizeof(float));
      (*news_dict)[nid_addr[k]] = ret;
      outputs0_addr += 400;
    }
    startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    model0_cost_time.insert(std::pair<double, double>(startTime_ms, endTime_ms));
  }
  std::cout << "=========InitNewsDict end" << std::endl;
  return;
}

void InitUserDict(const std::vector<std::string> &history_input_files, const std::map<uint32_t, MSTensor> &news_dict,
                  std::map<uint32_t, MSTensor> *user_dict, Model *model1) {
  int count = 0;
  int uid_count = 0;
  for (size_t i = 0; i < history_input_files.size() - 1; ++i) {
    // model1, get model1 outputs
    struct timeval start;
    struct timeval end;
    double startTime_ms;
    double endTime_ms;
    std::vector<MSTensor> outputs1;
    int64_t size = FLAGS_batch_size * 50 * 400 * sizeof(float);
    MSTensor buffer("", mindspore::DataType::kNumberTypeFloat32, {FLAGS_batch_size, 50, 400}, nullptr, size);
    uint8_t *addr = reinterpret_cast<uint8_t *>(buffer.MutableData());
    auto file_name = FLAGS_base_path + "/users_test_data/01_history_data/naml_users_" +
                     std::to_string(count++) + ".bin";
    auto nid = ReadFileToTensor(file_name);
    auto nid_addr = reinterpret_cast<uint32_t *>(nid.MutableData());

    for (int j = 0; j < FLAGS_batch_size; ++j) {
      for (int k = 0; k < 50; ++k) {
        if (news_dict.find(nid_addr[k]) == news_dict.end()) {
          addr += 400 * sizeof(float);
          continue;
        }
        auto ms_tensor = news_dict.at(nid_addr[k]);
        uint32_t *new_dict_data = reinterpret_cast<uint32_t *>(ms_tensor.MutableData());
        if (addr == nullptr || new_dict_data == nullptr) {
          std::cout << "addr is nullptr or new_dict_data is nullptr"
                    << "src is: " << &new_dict_data << " dst is: " << &addr << std::endl;
          return;
        }
        memcpy(addr, new_dict_data, 400 * sizeof(float));
        addr += 400 * sizeof(float);
      }
      nid_addr += 50;
    }
    gettimeofday(&start, nullptr);
    Status ret1 = model1->Predict({buffer}, &outputs1);
    gettimeofday(&end, nullptr);
    if (ret1 != kSuccess) {
      std::cout << "ERROR: Predict model1 failed." << std::endl;
      return;
    }
    // init user_dict
    auto file_name1 = FLAGS_base_path + "/users_test_data/00_user_id_data/naml_users_" +
                      std::to_string(uid_count++) + ".bin";
    auto nid1 = ReadFileToTensor(file_name1);
    auto nid_addr1 = reinterpret_cast<uint32_t *>(nid1.MutableData());
    auto outputs1_addr = reinterpret_cast<float *>(outputs1[0].MutableData());
    for (int k = 0; k < FLAGS_batch_size; ++k) {
      MSTensor ret("", mindspore::DataType::kNumberTypeFloat32, {400}, nullptr, sizeof(float) * 400);
      uint32_t *addr1 = reinterpret_cast<uint32_t *>(ret.MutableData());
      memcpy(addr1, outputs1_addr, 400 * sizeof(float));
      addr1 += 400;
      (*user_dict)[nid_addr1[k]] = ret;
      outputs1_addr += 400;
    }
    startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    model1_cost_time.insert(std::pair<double, double>(startTime_ms, endTime_ms));
  }
  std::cout << "=========InitUserDict end" << std::endl;
  return;
}

void InitPred(const std::map<uint32_t, MSTensor> &news_dict, const std::map<uint32_t, MSTensor> &user_dict) {
  std::string browsed_news_path = FLAGS_base_path + "/browsed_news_test_data";
  std::string file1_path = browsed_news_path + "/01_candidate_nid_data";
  auto eval_candidate_news = GetAllFiles(file1_path);
  for (size_t i = 0; i < eval_candidate_news.size() - 1; ++i) {
    std::vector<MSTensor> pred;
    std::string file2_path = browsed_news_path + "/00_user_id_data";
    auto uid_file = file2_path + "/naml_browsed_news_" + std::to_string(i) + ".bin";
    auto uid_nid = ReadFileToTensor(uid_file);
    auto uid_nid_addr = reinterpret_cast<uint32_t *>(uid_nid.MutableData());
    if (user_dict.find(uid_nid_addr[0]) == user_dict.end()) {
      int rst = WriteResult(uid_file, pred);
      if (rst != 0) {
        std::cout << "write result failed." << std::endl;
        return;
      }
      continue;
    }

    MSTensor dot2 = user_dict.at(uid_nid_addr[0]);
    auto candidate_nid_file = file1_path + "/naml_browsed_news_" + std::to_string(i) + ".bin";
    auto candidate_nid = ReadFileToTensor(candidate_nid_file);
    size_t bin_size = candidate_nid.DataSize();
    size_t browsed_news_count = bin_size / sizeof(float);

    auto candidate_nid_addr = reinterpret_cast<uint32_t *>(candidate_nid.MutableData());
    MSTensor ret("", mindspore::DataType::kNumberTypeFloat32, {static_cast<int>(browsed_news_count)},
                 nullptr, bin_size);
    uint8_t *addr = reinterpret_cast<uint8_t *>(ret.MutableData());
    for (size_t j = 0; j < browsed_news_count; ++j) {
      float sum = 0;
      if (news_dict.find(candidate_nid_addr[j]) == news_dict.end()) {
        addr += sizeof(float);
        continue;
      }

      MSTensor dot1 = news_dict.at(candidate_nid_addr[j]);
      auto dot1_addr = reinterpret_cast<float *>(dot1.MutableData());
      auto dot2_addr = reinterpret_cast<float *>(dot2.MutableData());
      for (int k = 0; k < 400; ++k) {
        sum = sum + dot1_addr[k] * dot2_addr[k];
      }
      memcpy(addr, &sum, sizeof(float));
      addr += sizeof(float);
    }
    pred.emplace_back(ret);
    int rst = WriteResult(uid_file, pred);
    if (rst != 0) {
      std::cout << "write result failed." << std::endl;
      return;
    }
  }
  std::cout << "=========InitPred end" << std::endl;
  return;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (RealPath(FLAGS_news_mindir).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  context->MutableDeviceInfo().push_back(ascend310);

  double startTimeMs;
  double endTimeMs;
  std::map<double, double> total_cost_time;
  // om -> model
  Model model0;
  Model model1;
  if (InitModel(FLAGS_news_mindir, &model0) != 0 || InitModel(FLAGS_user_mindir, &model1) != 0) {
    std::cout << "ERROR: Init model failed." << std::endl;
    return 1;
  }

  // get model inputs
  std::vector<MSTensor> model0_inputs = model0.GetInputs();
  if (model0_inputs.empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }

  // get input files by bin files
  std::string news_dataset_path = FLAGS_base_path + "/news_test_data";
  std::string history_data_path = FLAGS_base_path + "/users_test_data/01_history_data";
  news_input0_0_files = GetAllFiles(news_dataset_path + "/00_category_data");
  auto history_input_files = GetAllFiles(history_data_path);
  if (news_input0_0_files.empty() || history_input_files.empty()) {
    std::cout << "ERROR: input data empty." << std::endl;
    return 1;
  }

  std::map<uint32_t, MSTensor> news_dict;
  std::map<uint32_t, MSTensor> user_dict;

  InitNewsDict(news_input0_0_files, model0_inputs, &news_dict, &model0);
  InitUserDict(history_input_files, news_dict, &user_dict, &model1);
  InitPred(news_dict, user_dict);
  gettimeofday(&total_end, nullptr);

  startTimeMs = (1.0 * total_start.tv_sec * 1000000 + total_start.tv_usec) / 1000;
  endTimeMs = (1.0 * total_end.tv_sec * 1000000 + total_end.tv_usec) / 1000;
  total_cost_time.insert(std::pair<double, double>(startTimeMs, endTimeMs));

  double average0 = 0.0;
  int inferCount0 = 0;
  double average1 = 0.0;
  int infer_cnt = 0;
  GetAveTime(model0_cost_time, &inferCount0, &average0);
  GetAveTime(model1_cost_time, &infer_cnt, &average1);
  double total_time = total_cost_time.begin()->second - total_cost_time.begin()->first;

  std::stringstream timeCost0, timeCost1, totalTimeCost;
  timeCost0 << "first model cost average time: "<< average0 << " ms of infer_count " << inferCount0 << std::endl;
  timeCost1 << "second model cost average time: "<< average1 << " ms of infer_count " << infer_cnt << std::endl;
  totalTimeCost << "total cost time: "<< total_time << " ms of infer_count " << infer_cnt << std::endl;

  std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
  fileStream << timeCost0.str() << std::endl;
  fileStream << timeCost1.str() << std::endl;
  fileStream << totalTimeCost.str() << std::endl;
  std::cout << timeCost0.str() << std::endl;
  std::cout << timeCost1.str() << std::endl;
  std::cout << totalTimeCost.str() << std::endl;
  fileStream.close();
  model0_cost_time.clear();
  model1_cost_time.clear();
  total_cost_time.clear();
  std::cout << "Execute success." << std::endl;
  return 0;
}

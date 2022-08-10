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
#include <cstring>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"

#include "inc/utils.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;

int load_model(Model *model, std::vector<MSTensor> *model_inputs, std::string mindir_path, int device_id);

int main(int argc, char **argv) {
  std::string sdne_file(argv[1]);
  std::string dataset_name(argv[2]);
  std::string graph_file(argv[3]);
  std::string device_id_str(argv[4]);
  int device_id = stoi(device_id_str);
  Model model;
  std::vector<MSTensor> model_inputs;
  load_model(&model, &model_inputs, sdne_file, device_id);
  auto graph = GetGraph(graph_file, dataset_name != "WIKI");
  std::vector< std::vector<float> > embeddings;
  std::vector< std::vector<float> > reconstructions;

  std::map<double, double> costTime_map;
  struct timeval start = {0};
  struct timeval end = {0};

  for (unsigned int i = 0; i < graph.size(); i++) {
    std::vector<MSTensor> outputs;
    auto inp_shape = model_inputs[0].Shape();
    auto inp_data = GetDataFromGraph(graph[i], inp_shape[1]);
    double startTimeMs;
    double endTimeMs;
    memcpy(reinterpret_cast<float*>(model_inputs[0].MutableData()), &inp_data[0], inp_shape[1] * sizeof(float));
    gettimeofday(&start, nullptr);
    Status ret = model.Predict(model_inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Calculate node " << i << " failed." << std::endl;
      return 1;
    }
    float *pdata = reinterpret_cast<float*>(outputs[1].MutableData());
    auto emb_shape = outputs[1].Shape();
    embeddings.push_back(Tensor2Vector(pdata, emb_shape[1]));
    pdata = reinterpret_cast<float*>(outputs[0].MutableData());
    auto rec_shape = outputs[0].Shape();
    reconstructions.push_back(Tensor2Vector(pdata, rec_shape[1]));

    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
  }

  WriteResult(embeddings, reconstructions);

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

int load_model(Model *model, std::vector<MSTensor> *model_inputs, std::string mindir_path, int device_id) {
  if (RealPath(mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(device_id);
  context->MutableDeviceInfo().push_back(ascend310);
  mindspore::Graph graph;
  Serialization::Load(mindir_path, ModelType::kMindIR, &graph);

  Status ret = model->Build(GraphCell(graph), context);
  if (ret != kSuccess) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }

  *model_inputs = model->GetInputs();
  if (model_inputs->empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }
  return 0;
}

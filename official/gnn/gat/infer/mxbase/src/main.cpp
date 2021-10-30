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
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "GatNerBase.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;
uint32_t g_TP = 0;
uint32_t g_FP = 0;
uint32_t g_FN = 0;
uint32_t NODES = 3312;
uint32_t FEATURES = 3703;
uint32_t CLASS_NUM = 6;

void InitGatParam(InitParam* initParam) {
  initParam->deviceId = 0;
  initParam->labelPath = "../data/config/infer_label.txt";
  initParam->modelPath = "../data/model/citeseer.om";
  initParam->classNum = 6;
}

int main(int argc, char* argv[]) {
  if (argc <= 5) {
    LogWarn << "Please input data path, model path, node num, feature num, "
               "class num.";
    return APP_ERR_OK;
  }

  InitParam initParam;
  InitGatParam(&initParam);
  initParam.modelPath = argv[3];
  NODES = atoi(argv[4]);
  FEATURES = atoi(argv[5]);
  CLASS_NUM = atoi(argv[6]);
  auto gatBase = std::make_shared<GatNerBase>();
  APP_ERROR ret = gatBase->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "Gatbase init failed, ret=" << ret << ".";
    return ret;
  }

  std::string inferPath = argv[1];
  std::vector<std::string> files;
  files.push_back(argv[1]);

  // do eval and calc the f1 score
  bool eval = atoi(argv[2]);
  for (uint32_t i = 0; i < files.size(); i++) {
    LogInfo << "read file name: " << files[i];
    ret = gatBase->Process(inferPath, files[i], eval);
    if (ret != APP_ERR_OK) {
      LogError << "Gatbase process failed, ret=" << ret << ".";
      gatBase->DeInit();
      return ret;
    }
  }

  if (eval) {
    LogInfo << "==============================================================";
    float precision = g_TP * 1.0 / (g_TP + g_FP);
    LogInfo << "Precision: " << precision;
    LogInfo << "==============================================================";
  }
  gatBase->DeInit();
  double costSum = 0;
  for (uint32_t i = 0; i < g_inferCost.size(); i++) {
    costSum += g_inferCost[i];
  }
  LogInfo << "Infer images sum " << g_inferCost.size()
          << ", cost total time: " << costSum << " ms.";
  LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum
          << " bin/sec.";
  return APP_ERR_OK;
}

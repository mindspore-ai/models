/*
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <unistd.h>
#include <iostream>
#include <fstream>
#include "MxBase/Log/Log.h"
#include "Hourglass.h"

namespace {
const uint32_t DEVICE_ID = 0;
const uint32_t FILE_NUM = 3258;
}  // namespace


int main(int argc, char *argv[]) {
  if (argc <= 2) {
    LogWarn << "Please input image path, such as './build/hourglass "
               "[om_file_path] [inferPath]' ";
    return APP_ERR_OK;
  }

  InitParam initParam = {};
  initParam.deviceId = DEVICE_ID;
  initParam.checkTensor = false;
  initParam.modelPath = argv[1];

  auto inferHourglass = std::make_shared<Hourglass>();
  APP_ERROR ret = inferHourglass->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "Hourglass init failed, ret=" << ret << ".";
    return ret;
  }
  std::string inferPath = argv[2];
  for (uint32_t i = 0; i < FILE_NUM; i++) {
      std::string fileName = "data" + std::to_string(i) + ".bin";
      LogInfo << "read file name: " << fileName;
      ret = inferHourglass->Process(inferPath, fileName);
      if (ret != APP_ERR_OK) {
        LogError << "Hourglass process failed, ret=" << ret << ".";
        inferHourglass->DeInit();
        return ret;
      }
  }
  eval();
  inferHourglass->DeInit();
  return APP_ERR_OK;
}


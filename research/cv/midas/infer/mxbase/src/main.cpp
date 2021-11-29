/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
/* mxbase main */
#include "Midas.h"
#include "MxBase/Log/Log.h"

namespace {
const uint32_t DEVICE_ID = 0;
const char RESULT_PATH[] = "../data/mx_result/";

}  // namespace

int main(int argc, char *argv[]) {
  int tmp = 3;
  if (argc <= tmp) {
    LogWarn << "Please input image path, such as './ [om_file_path] [img_path] "
               "[dataset_name]'.";
    return APP_ERR_OK;
  }
  InitParam initParam = {};
  initParam.deviceId = DEVICE_ID;

  initParam.checkTensor = true;

  initParam.modelPath = argv[1];
  auto inferMidas = std::make_shared<Midas>();
  APP_ERROR ret = inferMidas->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "Midas init failed, ret=" << ret << ".";
    return ret;
  }
  std::string imgPath = argv[2];
  std::string dataset_name = argv[3];
  ret = inferMidas->Process(imgPath, RESULT_PATH, dataset_name);
  if (ret != APP_ERR_OK) {
    LogError << "Midas process failed, ret=" << ret << ".";
    inferMidas->DeInit();
    return ret;
  }
  inferMidas->DeInit();
  return APP_ERR_OK;
}

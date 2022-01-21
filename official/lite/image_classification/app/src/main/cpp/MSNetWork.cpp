/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "MSNetWork.h"
#include <android/log.h>
#include <iostream>
#include <string>
#include "include/errorcode.h"

#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)

MSNetWork::MSNetWork(void) : model_(nullptr) {}

MSNetWork::~MSNetWork(void) {}

bool MSNetWork::BuildModel(char *modelBuffer, size_t bufferLen,
                           std::shared_ptr<mindspore::Context> ctx) {
  model_ = std::make_shared<mindspore::Model>();
  if (model_ == nullptr) {
    MS_PRINT("MindSpore build model failed!.");
    return false;
  }
  auto ret = model_->Build(modelBuffer, bufferLen, mindspore::ModelType::kMindIR, ctx);
  return ret.IsOk();
}

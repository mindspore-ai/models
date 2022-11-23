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
#ifndef MINDSPORE_MODELS_UTILS_INFER_H_
#define MINDSPORE_MODELS_UTILS_INFER_H_

#include <string>
#include <memory>
#include "common_inc/utils.h"
#include "common_inc/flags.h"
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/dataset/execute.h"
#include "include/dataset/transforms.h"
#ifdef ENABLE_LITE
#include "include/dataset/vision_lite.h"
#else
#include "include/dataset/vision.h"
#endif
#include "include/dataset/vision_ascend.h"

using mindspore::Context;
using mindspore::DataType;
using mindspore::kSuccess;
using mindspore::Model;
using mindspore::ModelType;
using mindspore::MSTensor;
using mindspore::Serialization;
using mindspore::Status;
using mindspore::GraphCell;

using mindspore::dataset::Execute;
using mindspore::dataset::InterpolationMode;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::transforms::TypeCast;
using namespace mindspore::dataset::vision;  // NOLINT

bool LoadModel(const std::string &mindir_path, const std::shared_ptr<mindspore::Context> &context, Model *model) {
  if (!model || !context) {
    return false;
  }
  if (RealPath(mindir_path).empty()) {
    std::cout << "Model path cannot be empty" << std::endl;
    return false;
  }
#ifdef ENABLE_LITE
  auto ret = model->Build(mindir_path, mindspore::kMindIR, context);
#else
  mindspore::Graph graph;
  Serialization::Load(mindir_path, mindspore::kMindIR, &graph);
  auto ret = model->Build(GraphCell(graph), context);
#endif
  if (ret != kSuccess) {
    std::cout << "ERROR: Build failed." << std::endl;
    return false;
  }
  auto model_inputs = model->GetInputs();
  if (model_inputs.empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return false;
  }
  return true;
}

bool LoadModel(const std::string &mindir_path, const std::string &device_type, uint32_t device_id,
               const std::shared_ptr<mindspore::AscendDeviceInfo> &ascend_device_info, Model *model) {
  if (!model || !ascend_device_info) {
    return false;
  }
  auto context = std::make_shared<mindspore::Context>();
  if (device_type == "Ascend") {
    ascend_device_info->SetDeviceID(device_id);
    context->MutableDeviceInfo().push_back(ascend_device_info);
  } else if (device_type == "GPU") {
    auto device_info = std::make_shared<mindspore::GPUDeviceInfo>();
    device_info->SetDeviceID(device_id);
    context->MutableDeviceInfo().push_back(device_info);
  } else if (device_type == "CPU") {
    auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
    context->MutableDeviceInfo().push_back(device_info);
  } else {
    std::cout << "Invalid device type " << device_type << std::endl;
    return false;
  }
  return LoadModel(mindir_path, context, model);
}

bool LoadModel(const std::string &mindir_path, const std::string &device_type, uint32_t device_id, Model *model) {
  if (!model) {
    return false;
  }
  auto context = std::make_shared<mindspore::Context>();
  if (device_type == "Ascend") {
    auto ascend = std::make_shared<mindspore::AscendDeviceInfo>();
    ascend->SetDeviceID(device_id);
    context->MutableDeviceInfo().push_back(ascend);
  } else if (device_type == "GPU") {
    auto device_info = std::make_shared<mindspore::GPUDeviceInfo>();
    device_info->SetDeviceID(device_id);
    context->MutableDeviceInfo().push_back(device_info);
  } else if (device_type == "CPU") {
    auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
    context->MutableDeviceInfo().push_back(device_info);
  } else {
    std::cout << "Invalid device type " << device_type << std::endl;
    return false;
  }
  return LoadModel(mindir_path, context, model);
}
#endif  // MINDSPORE_MODELS_UTILS_INFER_H_

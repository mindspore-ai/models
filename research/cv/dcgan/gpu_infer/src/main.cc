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
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>

#include "../inc/utils.h"
#include "include/dataset/execute.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision.h"
#include "include/api/types.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using mindspore::Serialization;
using mindspore::Model;
using mindspore::Context;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::Graph;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::DataType;
using mindspore::dataset::Execute;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::Decode;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::HWC2CHW;

DEFINE_string(mindir_path, "", "model path");
DEFINE_string(dataset_path, "", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(mode, "", "train or test");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Model model;

    std::vector<MSTensor> model_inputs;
    if (RealPath(FLAGS_mindir_path).empty()) {
        std::cout << "Invalid mindir" << std::endl;
        return -1;
    }

    auto context = std::make_shared<Context>();
    auto gpu_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
    gpu_device_info->SetDeviceID(FLAGS_device_id);
    context->MutableDeviceInfo().push_back(gpu_device_info);
    mindspore::Graph graph;
    Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);

    Status ret_build = model.Build(GraphCell(graph), context);
    if (ret_build != kSuccess) {
        std::cout << "ERROR: Build failed." << std::endl;
        return -1;
    }

    model_inputs = model.GetInputs();
    if (model_inputs.empty()) {
        std::cout << "Invalid model, inputs is empty." << std::endl;
        return -1;
    }

    auto input0_files = GetAllFiles(FLAGS_dataset_path);
    if (input0_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }
    size_t size = input0_files.size();
    for (size_t i = 0; i < size; ++i) {
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files:" << input0_files[i] <<std::endl;
        auto input0 = ReadFileToTensor(input0_files[i]);
        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            input0.Data().get(), input0.DataSize());

        Status ret_predict = model.Predict(inputs, &outputs);
        if (ret_predict != kSuccess) {
            std::cout << "Predict " << input0_files[i] << " failed." << std::endl;
            return 1;
        }

        int rst = WriteResult(input0_files[i], outputs, FLAGS_mode);
        if (rst != 0) {
            std::cout << "write result failed." << std::endl;
            return rst;
        }
    }

    return 0;
}

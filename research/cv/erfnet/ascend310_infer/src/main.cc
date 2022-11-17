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
#include <sys/stat.h>
#include <sys/time.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"
#include "inc/utils.h"

namespace ms = mindspore;
namespace ds = mindspore::dataset;

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "example: ./erfnet /path/to/model /path/to/image device_id " << std::endl;
        return -1;
    }
    std::cout << "model_patt:" << argv[1] << std::endl;
    std::cout << "image_path:" << argv[2] << std::endl;
    std::cout << "result_path" << argv[3] << std::endl;
    std::cout << "device_id:" << argv[4] << std::endl;
    int device_id = argv[4][0] - '0';
    std::string res_path = argv[3];

    // set context
    auto context = std::make_shared<ms::Context>();
    auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(device_id);
    context->MutableDeviceInfo().push_back(ascend310_info);

    // define model
    ms::Graph graph;
    ms::Status ret = ms::Serialization::Load(argv[1], ms::ModelType::kMindIR, &graph);
    if (ret != ms::kSuccess) {
        std::cout << "Load model failed." << std::endl;
        return 1;
    }
    ms::Model erfnet;

    // build model
    ret = erfnet.Build(ms::GraphCell(graph), context);
    if (ret != ms::kSuccess) {
        std::cout << "Build model failed." << std::endl;
        return 1;
    }

    // get model info
    std::vector<ms::MSTensor> model_inputs = erfnet.GetInputs();
    if (model_inputs.empty()) {
        std::cout << "Invalid model, inputs is empty." << std::endl;
        return 1;
    }

    // define transforms
    std::shared_ptr<ds::TensorTransform> decode(new ds::vision::Decode());
    std::shared_ptr<ds::TensorTransform> resize(new ds::vision::Resize({512, 1024}));
    std::shared_ptr<ds::TensorTransform> normalize(new ds::vision::Normalize({0, 0, 0},
                                                                             {255, 255, 255}));
    std::shared_ptr<ds::TensorTransform> hwc2chw(new ds::vision::HWC2CHW());

    // define preprocessor
    ds::Execute preprocessor({decode, resize, normalize, hwc2chw});

    std::map<double, double> costTime_map;

    std::vector<std::string> images = GetAllFiles(argv[2]);
    for (const auto &image_file : images) {
        struct timeval start = {0};
        struct timeval end = {0};
        double startTime_ms;
        double endTime_ms;

        // prepare input
        std::vector<ms::MSTensor> outputs;
        std::vector<ms::MSTensor> inputs;

        // read image file and preprocess
        auto image = ReadFile(image_file);
        ret = preprocessor(image, &image);
        if (ret != ms::kSuccess) {
            std::cout << "Image preprocess failed." << std::endl;
            return 1;
        }

        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            image.Data().get(), image.DataSize());

        // infer
        gettimeofday(&start, NULL);
        ret = erfnet.Predict(inputs, &outputs);
        gettimeofday(&end, NULL);
        if (ret != ms::kSuccess) {
            std::cout << "Predict model failed." << std::endl;
            return 1;
        }

        // print infer result
        std::cout << "Image: " << image_file << std::endl;
        WriteResult(image_file, outputs, res_path);
        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
    }

    double average = 0.0;
    int infer_cnt = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
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

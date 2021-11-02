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
#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iosfwd>

#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"
#include "include/api/types.h"


DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(input0_path, ".", "input0 path");
DEFINE_int32(device_id, 0, "device id");


using mindspore::kSuccess;
using mindspore::MSTensor;

namespace ms = mindspore;
namespace ds = mindspore::dataset;


std::vector<std::string> GetAllFiles(std::string_view dir_name);
DIR* OpenDir(std::string_view dir_name);
std::string RealPath(std::string_view path);
ms::MSTensor ReadFile(const std::string& file);
mindspore::MSTensor ReadFileToTensor(const std::string& file);
int WriteResult(const std::string& imageFile, const std::vector<MSTensor>& outputs);


int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (RealPath(FLAGS_mindir_path).empty()) {
        std::cout << "Invalid mindir" << std::endl;
        return 1;
    }

    // set context
    auto context = std::make_shared<ms::Context>();
    auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(FLAGS_device_id);
    context->MutableDeviceInfo().push_back(ascend310_info);

    // define model
    ms::Graph graph;
    ms::Status ret = ms::Serialization::Load(FLAGS_mindir_path, ms::ModelType::kMindIR, &graph);
    if (ret != ms::kSuccess) {
        std::cout << "Load model failed." << std::endl;
        return 1;
    }
    ms::Model gan;

    // build model
    ret = gan.Build(ms::GraphCell(graph), context);
    if (ret != ms::kSuccess) {
        std::cout << "Build model failed." << std::endl;
        return 1;
    }

    // get model info
    std::vector<ms::MSTensor> model_inputs = gan.GetInputs();
    if (model_inputs.empty()) {
        std::cout << "Invalid model, inputs is empty." << std::endl;
        return 1;
    }

    auto input0_files = GetAllFiles(FLAGS_input0_path);
    if (input0_files.empty()) {
        std::cout << "ERROR: input data empty." << std::endl;
    }

    size_t size = input0_files.size();

    for (size_t i = 0; i < size; i++) {
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files:" << input0_files[i] << std::endl;

        auto input0 = ReadFileToTensor(input0_files[i]);

        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
            input0.Data().get(), input0.DataSize());

        // infer
        ret = gan.Predict(inputs, &outputs);
        if (ret != kSuccess) {
            std::cout << "Predict " << input0_files[i] << "failed." << std::endl;
            return 1;
        }
        WriteResult(input0_files[i], outputs);
    }

    return 0;
}

std::vector<std::string> GetAllFiles(std::string_view dir_name) {
    struct dirent* filename;
    DIR* dir = OpenDir(dir_name);
    if (dir == nullptr) {
        return {};
    }

    /* read all the files in the dir ~ */
    std::vector<std::string> res;
    while ((filename = readdir(dir)) != nullptr) {
        std::string d_name = std::string(filename->d_name);
        // get rid of "." and ".."
        if (d_name == "." || d_name == ".." || filename->d_type != DT_REG)
            continue;
        res.emplace_back(std::string(dir_name) + "/" + filename->d_name);
    }

    std::sort(res.begin(), res.end());
    return res;
}

DIR* OpenDir(std::string_view dir_name) {
    // check the parameter !
    if (dir_name.empty()) {
        std::cout << " dir_name is null ! " << std::endl;
        return nullptr;
    }

    std::string real_path = RealPath(dir_name);

    // check if dir_name is a valid dir
    struct stat s;
    lstat(real_path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        std::cout << "dir_name is not a valid directory !" << std::endl;
        return nullptr;
    }

    DIR* dir;
    dir = opendir(real_path.c_str());
    if (dir == nullptr) {
        std::cout << "Can not open dir! " << dir_name << std::endl;
        return nullptr;
    }
    return dir;
}

std::string RealPath(std::string_view path) {
    char real_path_mem[PATH_MAX] = { 0 };
    char* real_path_ret = realpath(path.data(), real_path_mem);

    if (real_path_ret == nullptr) {
        std::cout << "File: " << path << " is not exist.";
        return "";
    }

    return std::string(real_path_mem);
}

ms::MSTensor ReadFile(const std::string& file) {
    if (file.empty()) {
        std::cout << "Pointer file is nullptr !" << std::endl;
        return ms::MSTensor();
    }

    std::ifstream ifs(file);
    if (!ifs.good()) {
        std::cout << "File: " << file << " is not exist !" << std::endl;
        return ms::MSTensor();
    }

    if (!ifs.is_open()) {
        std::cout << "File: " << file << "open failed !" << std::endl;
        return ms::MSTensor();
    }

    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ms::MSTensor buffer(file, ms::DataType::kNumberTypeUInt8, { static_cast<int64_t>(size) }, nullptr, size);

    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(buffer.MutableData()), size);
    ifs.close();

    return buffer;
}

mindspore::MSTensor ReadFileToTensor(const std::string& file) {
    if (file.empty()) {
        std::cout << "Pointer file is nullptr !" << std::endl;
        return mindspore::MSTensor();
    }

    std::ifstream ifs(file);
    if (!ifs.good()) {
        std::cout << "File: " << file << " is not exist !" << std::endl;
        return mindspore::MSTensor();
    }

    if (!ifs.is_open()) {
        std::cout << "File: " << file << "open failed !" << std::endl;
        return mindspore::MSTensor();
    }

    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    mindspore::MSTensor buffer(file, mindspore::DataType::kNumberTypeUInt8, {
        static_cast<int64_t>(size) }, nullptr, size);

    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(buffer.MutableData()), size);
    ifs.close();

    return buffer;
}


int WriteResult(const std::string& imageFile, const std::vector<MSTensor>& outputs) {
    std::string homePath = "./result_files";
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        std::shared_ptr<const void> netOutput;
        netOutput = outputs[i].Data();
        outputSize = outputs[i].DataSize();
        int pos = imageFile.rfind('/');
        std::string fileName(imageFile, pos + 1);
        fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), '_' + std::to_string(i) + ".bin");
        std::string outFileName = homePath + "/" + fileName;
        FILE* outputFile = fopen(outFileName.c_str(), "wb");
        fwrite(netOutput.get(), outputSize, sizeof(char), outputFile);
        fclose(outputFile);
        outputFile = nullptr;
    }
    return 0;
}

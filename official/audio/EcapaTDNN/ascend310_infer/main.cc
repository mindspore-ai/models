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

#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"

uint64_t GetTimeMicroSeconds() {
    struct timespec t;
    t.tv_sec = t.tv_nsec = 0;
    clock_gettime(/*CLOCK_REALTIME*/0, &t);
    return (uint64_t)t.tv_sec * 1000000ULL + t.tv_nsec / 1000L;
}
struct stat info;
namespace ms = mindspore;
namespace ds = mindspore::dataset;

std::vector<std::string> GetAllFiles(std::string_view dir_name);
DIR *OpenDir(std::string_view dir_name);
std::string RealPath(std::string_view path);
size_t WriteFile(ms::MSTensor& data, std::string outfile);
ms::MSTensor ReadFile(const std::string &file);

int main(int argc, char **argv) {
    // set context
    auto context = std::make_shared<ms::Context>();
    auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(0);
    context->MutableDeviceInfo().push_back(ascend310_info);
    std::string ecapa_file = argv[1];
    std::string image_path = argv[2];
    std::string out_path = argv[3];
    out_path = out_path + "/";
    // define model
    ms::Graph graph;
    ms::Status ret = ms::Serialization::Load(ecapa_file, ms::ModelType::kMindIR, &graph);
    if (ret != ms::kSuccess) {
      std::cout << "Load model failed." << std::endl;
      return 1;
    }
    std::cout << "Load model success." << std::endl;
    ms::Model ecapatdnn;

    // build model
    ret = ecapatdnn.Build(ms::GraphCell(graph), context);
    if (ret != ms::kSuccess) {
      std::cout << "Build model failed." << std::endl;
      return 1;
    }
    std::cout << "Build model success." << std::endl;
    // get model info
    std::vector<ms::MSTensor> model_inputs = ecapatdnn.GetInputs();
    if (model_inputs.empty()) {
      std::cout << "Invalid model, inputs is empty." << std::endl;
      return 1;
    }

    std::string flistname = out_path + "emb.txt";
    std::ofstream fpout(flistname);
    std::vector<std::string> feats = GetAllFiles(image_path);
    uint64_t Time1 = GetTimeMicroSeconds();
    for (const auto &feat_file : feats) {
        std::size_t found = feat_file.rfind("/");
        std::string fname = "";
        if (found != std::string::npos) {
            fname = feat_file.substr(found + 1);
        }
        // prepare input
        std::vector<ms::MSTensor> outputs;
        std::vector<ms::MSTensor> inputs;

        // read image file and preprocess
        auto feat = ReadFile(feat_file);
        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            feat.Data().get(), feat.DataSize());
        // infer
        ret = ecapatdnn.Predict(inputs, &outputs);
        if (ret != ms::kSuccess) {
            std::cout << "Predict model failed." << std::endl;
            return 1;
        }
        std::string outname = out_path + fname;
        fpout << fname << std::endl;
        WriteFile(outputs[0], outname);
    }
    uint64_t end = GetTimeMicroSeconds();
    printf("The total run time is: %f ms \n", static_cast<double>(end - Time1) / 1000);
    fpout.close();
    return 0;
}

std::vector<std::string> GetAllFiles(std::string_view dir_name) {
  struct dirent *filename;
  DIR *dir = OpenDir(dir_name);
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

DIR *OpenDir(std::string_view dir_name) {
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

  DIR *dir;
  dir = opendir(real_path.c_str());
  if (dir == nullptr) {
    std::cout << "Can not open dir " << dir_name << std::endl;
    return nullptr;
  }
  return dir;
}

std::string RealPath(std::string_view path) {
  char real_path_mem[PATH_MAX] = {0};
  char *real_path_ret = realpath(path.data(), real_path_mem);

  if (real_path_ret == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  return std::string(real_path_mem);
}

ms::MSTensor ReadFile(const std::string &file) {
  if (file.empty()) {
    std::cout << "Pointer file is nullptr" << std::endl;
    return ms::MSTensor();
  }

  std::ifstream ifs(file);
  if (!ifs.good()) {
    std::cout << "File: " << file << " is not exist" << std::endl;
    return ms::MSTensor();
  }

  if (!ifs.is_open()) {
    std::cout << "File: " << file << "open failed" << std::endl;
    return ms::MSTensor();
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  ms::MSTensor buffer(file, ms::DataType::kNumberTypeFloat32, {1, 301, 80}, nullptr, size);

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
  ifs.close();

  return buffer;
}

size_t WriteFile(ms::MSTensor& data, std::string outfile) {
    std::ofstream fout(outfile, std::ios::out | std::ios::binary);
    fout.write(reinterpret_cast<const char *>(data.MutableData()), data.DataSize());
    fout.close();
  return 0;
}

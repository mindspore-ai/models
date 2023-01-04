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

#include "../include/utils.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>


using mindspore::MSTensor;
using mindspore::DataType;


void read_points_num(const std::string &data_dir, int32_t &var) {
    var = 0;
    std::string path = data_dir + "/points_num.bin";
    std::ifstream fp(path.c_str(), std::ios::binary);

    while (fp.read(reinterpret_cast<char*>(&var), sizeof(int32_t))) {
    }
    fp.close();
}


void read_input(const std::string &data_dir, std::vector<float> &vec) {
    vec.clear();
    std::string path = data_dir + "/input.bin";
    std::ifstream fp(path);

    if (fp.is_open()) {
      std::string line;
      while (std::getline(fp, line)) {
          float var = static_cast<float>(atof(line.c_str()));
          char mem[PATH_MAX] = {'\0'};
          snprintf(mem, PATH_MAX, "%.16f", var);
          vec.push_back(var);
      }
      fp.close();
    }
}


void read_gt(const std::string &data_dir, std::vector<uint32_t> &vec) {
    vec.clear();
    std::string path = data_dir + "/gt.bin";
    std::ifstream fp(path.c_str(), std::ios::binary);

    uint32_t var = 0;
    while (fp.read(reinterpret_cast<char*>(&var), sizeof(uint32_t))) {
          vec.push_back(var);
    }
    fp.close();
}


std::string print_vec(const std::vector<float> &vec, size_t size = 50) {
    std::string var;
    for (size_t i = 0; i < size; ++i) {
        char mem[PATH_MAX] = {'\0'};
        snprintf(mem, PATH_MAX, "%.16f", vec[i]);
        std::string str(mem);
        var += str + ", ";
    }
    return var;
}


std::string print_vec(const std::vector<uint32_t> &vec, size_t size = 50) {
    std::string var;
    for (size_t i = 0; i < size; ++i) {
        var += std::to_string(vec[i]) + ", ";
    }

    return var;
}


std::string print_dataptr(const void *data_ptr, size_t size) {
    std::string var;
    const float *ptr = (const float *)data_ptr;
    for (size_t i = 0; i < size; ++i) {
        char mem[PATH_MAX] = {'\0'};
        snprintf(mem, PATH_MAX, "%.16f", ptr[i]);
        std::string str(mem);
        var += str + ", ";
    }
    return var;
}


std::string print_data_shape(const std::vector<int64_t> &shape) {
    std::string var;
    for (size_t i = 0; i < shape.size(); ++i) {
        var += std::to_string(shape[i]) + ", ";
    }
    return var;
}


void GetDataSample(const std::string& data_dir, int32_t &points_num,
                   std::vector<float> &input, std::vector<uint32_t> &gt) {
    read_points_num(data_dir, points_num);

    read_input(data_dir, input);

    read_gt(data_dir, gt);
}

static std::string RealPath(const std::string &path) {
  char realPathMem[PATH_MAX] = {0};
  char *realPathRet = realpath(path.data(), realPathMem);
  if (realPathRet == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }
  std::string realPath(realPathMem);
  return realPath;
}

static DIR *OpenDir(const std::string &dirName) {
  if (dirName.empty()) {
    std::cout << " dirName is null ! " << std::endl;
    return nullptr;
  }
  std::string realPath = RealPath(dirName);
  struct stat s;
  lstat(realPath.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dirName is not a valid directory !" << std::endl;
    return nullptr;
  }
  DIR *dir;
  dir = opendir(realPath.c_str());
  if (dir == nullptr) {
    std::cout << "Can not open dir " << dirName << std::endl;
    return nullptr;
  }
  std::cout << "Successfully opened the dir " << dirName << std::endl;
  return dir;
}

std::vector<std::string> GetDataDirs(const std::string& datasets_dir, const std::string& precision = "") {
    std::string dirName = datasets_dir + "/" + precision;
    std::cout << "dirName=" << dirName << std::endl;

    struct dirent *filename;
    DIR *dir = OpenDir(dirName);
    if (dir == nullptr) {
        return {};
    }

    std::vector<std::string> vec;
    while ((filename = readdir(dir)) != nullptr) {
        std::string dName = std::string(filename->d_name);
        if ((dName != ".") && (dName != "..") && (filename->d_type == DT_DIR)) {
            vec.emplace_back(std::string(dirName) + "/" + filename->d_name);
        }
    }

    std::sort(vec.begin(), vec.end());
    for (auto &item : vec) {
        std::cout << "data dir: " << item << std::endl;
    }
    return vec;
}


float custom_exp(double x) {
  return exp(x);
}


std::vector<float> softmax(const std::vector<float> &src) {
    double sum = 0;
    std::vector<float> dst(src);
    std::transform(src.begin(), src.end(), dst.begin(), custom_exp);
    sum = std::accumulate(dst.begin(), dst.end(), sum);
    for (size_t i = 0; i < src.size(); ++i) {
        dst.at(i) /= sum;
    }
    return dst;
}


float WriteResult(const std::string &data_dir, const std::vector<MSTensor> &outputs,
                  const std::vector<uint32_t> &gt, int32_t points_num) {
    // to vectors
    std::vector<std::vector<float>> vecs;
    size_t batch_num = outputs.size();
    size_t nodes_num = gt.size();

    for (size_t batch_idx = 0; batch_idx < batch_num; ++batch_idx) {
        const MSTensor &output = outputs[batch_idx];

        std::vector<int64_t> shape = output.Shape();

        std::shared_ptr<const void> sptrData = output.Data();
        const float *ptrData = (const float *)sptrData.get();


        for (size_t r = 0; r < size_t(shape[0]); ++r) {
            std::vector<float> vec;
            for (size_t c = 0; c < size_t(shape[1]); ++c) {
                size_t offset = r * size_t(shape[1]) + c;
                float prob = ptrData[offset];
                vec.push_back(prob);
            }
            if (vecs.size() >= nodes_num) {
                break; }
            vecs.push_back(vec);
        }
    }


    // softmax
    for (size_t r = 0; r < vecs.size(); ++r) {
        vecs[r] = softmax(vecs[r]);
    }

    // probs
    std::vector<float> probs;
    for (size_t r = 0; r < vecs.size(); ++r) {
        std::vector<float> &vec = vecs[r];
        size_t c_max = size_t(gt[r]);
        float prob = vec[c_max];
        probs.push_back(prob);
        if (prob == 0.0) {
            std::cout << "Error node: " << r << std::endl;}
    }

    // bpp
    float entropy = 0;
    for (const auto &prob : probs) {
      entropy += -log2(prob);
    }
    std::cout << "Points num: " << points_num << std::endl;
    float bpp = entropy / points_num;
    std::cout << "bpip: " << bpp << " bits for data_dir:" << data_dir << std::endl;

    return bpp;
}

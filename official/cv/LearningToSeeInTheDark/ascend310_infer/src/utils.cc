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

#include "inc/utils.h"

#include <fstream>
#include <algorithm>
#include <iostream>

using mindspore::MSTensor;
using mindspore::DataType;

std::vector<std::string> GetAllFiles(std::string dir_name) {
    struct dirent *filename;
    DIR *dir = OpenDir(dir_name);
    if (dir == nullptr) {
        return {};
    }
    std::vector<std::string> res;
    while ((filename = readdir(dir)) != nullptr) {
        std::string dName = std::string(filename->d_name);
        if (dName == "." || dName == ".." || filename->d_type != DT_REG) {
            continue;
        }
        res.emplace_back(std::string(dir_name) + "/" + filename->d_name);
    }
    std::sort(res.begin(), res.end());
    for (auto &f : res) {
        std::cout << "image file: " << f << std::endl;
    }
    return res;
}

int WriteResult(const std::string& imageFile, const std::vector<MSTensor> &outputs) {
    std::string homePath = "./result_Files";
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        std::shared_ptr<const void> netOutput = outputs[i].Data();
        outputSize = outputs[i].DataSize();
        int pos = imageFile.rfind('/');
        std::string fileName(imageFile, pos + 1);
        fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), '_' + std::to_string(i) + ".bin");
        std::string outFileName = homePath + "/" + fileName;
        FILE *outputFile = fopen(outFileName.c_str(), "wb");
        fwrite(netOutput.get(), outputSize, sizeof(char), outputFile);
        fclose(outputFile);
        outputFile = nullptr;
    }
    return 0;
}

mindspore::MSTensor ReadFileToTensor(const std::string &file) {
  if (file.empty()) {
    std::cout << "Pointer file is nullptr" << std::endl;
    return mindspore::MSTensor();
  }

  std::ifstream ifs(file);
  if (!ifs.good()) {
    std::cout << "File: " << file << " is not exist" << std::endl;
    return mindspore::MSTensor();
  }

  if (!ifs.is_open()) {
    std::cout << "File: " << file << "open failed" << std::endl;
    return mindspore::MSTensor();
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  mindspore::MSTensor buffer(file, mindspore::DataType::kNumberTypeUInt8, {static_cast<int64_t>(size)}, nullptr, size);

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
  ifs.close();

  return buffer;
}


DIR *OpenDir(std::string_view dir_name) {
    if (dir_name.empty()) {
        std::cout << " dir_name is null ! " << std::endl;
        return nullptr;
    }
    std::string real_path = RealPath(dir_name);
    struct stat s;
    lstat(real_path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        std::cout << "dir_name is not a valid directory !" << std::endl;
        return nullptr;
    }
    DIR *dir = opendir(real_path.c_str());
    if (dir == nullptr) {
        std::cout << "Can not open dir " << dir_name << std::endl;
        return nullptr;
    }
    std::cout << "Successfully opened the dir " << dir_name << std::endl;
    return dir;
}

std::string RealPath(std::string_view path) {
    char real_path_mem[PATH_MAX] = {0};
    char *real_path_ret = nullptr;
    real_path_ret = realpath(path.data(), real_path_mem);
    if (real_path_ret == nullptr) {
        std::cout << "File: " << path << " is not exist.";
        return "";
    }

    std::string real_path(real_path_mem);
    std::cout << path << " realpath is: " << real_path << std::endl;
    return real_path;
}


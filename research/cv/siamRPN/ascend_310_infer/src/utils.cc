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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
using mindspore::DataType;
using mindspore::MSTensor;

std::vector<std::string> GetAlldir(const std::string &dir_name, const std::string_view &data_name) {
  DIR *dir = OpenDir(dir_name + '/' + data_name.data());
  if (dir == nullptr) {
    return {};
  }
  std::vector<std::string> res;
  if (data_name == "vot2015" || data_name == "vot2016") {
    struct dirent *filename;
    while ((filename = readdir(dir)) != nullptr) {
      std::string d_name = std::string(filename->d_name);
      // get rid of "." and ".."
      if (d_name == "." || d_name == ".." || filename->d_type != DT_DIR)
        continue;
      std::cout << "dirs:" << d_name << std::endl;
      res.emplace_back(d_name);
    }
  }

  return res;
}

int WriteResult(const std::string &imageFile, float outputs[][4], int k,
                const std::string &dataset_name, const std::string &seq) {
  std::string homePath;
  homePath = "./result_Files/" + dataset_name + "/" + seq;
  std::string path = "mkdir ./result_Files/" + dataset_name;
  std::string path1 = "mkdir " + homePath;
  system(path.c_str());
  system(path1.c_str());
  std::cout << "homePath is " << homePath << std::endl;
  std::string fileName = homePath + '/' + imageFile;
  FILE *fp;
  fp = fopen(fileName.c_str(), "wt");
  for (int i = 0; i < k; i++) {
    fprintf(fp, "%f, ", outputs[i][0]);
    fprintf(fp, "%f, ", outputs[i][1]);
    fprintf(fp, "%f, ", outputs[i][2]);
    fprintf(fp, "%f\n", outputs[i][3]);
  }
  fclose(fp);
  return 0;
}

std::vector<std::string> GetAllFiles(std::string_view dirName) {
  struct dirent *filename;
  DIR *dir = OpenDir(dirName);
  if (dir == nullptr) {
    return {};
  }
  std::vector<std::string> res;
  while ((filename = readdir(dir)) != nullptr) {
    std::string dName = std::string(filename->d_name);
    if (dName == "." || dName == ".." || filename->d_type != DT_REG) {
      continue;
    }
    res.emplace_back(std::string(dirName) + "/" + filename->d_name);
  }
  std::sort(res.begin(), res.end());
  return res;
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
  mindspore::MSTensor buffer(file, mindspore::DataType::kNumberTypeUInt8,
                             {static_cast<int64_t>(size)}, nullptr, size);

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
  ifs.close();

  return buffer;
}

DIR *OpenDir(std::string_view dirName) {
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
  DIR *dir = opendir(realPath.c_str());
  if (dir == nullptr) {
    std::cout << "Can not open dir " << dirName << std::endl;
    return nullptr;
  }
  std::cout << "Successfully opened the dir " << dirName << std::endl;
  return dir;
}

std::string RealPath(std::string_view path) {
  char realPathMem[PATH_MAX] = {0};
  char *realPathRet = nullptr;
  realPathRet = realpath(path.data(), realPathMem);
  if (realPathRet == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  std::string realPath(realPathMem);
  std::cout << path << " realpath is: " << realPath << std::endl;
  return realPath;
}

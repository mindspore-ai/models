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

#ifndef MINDSPORE_MODELS_COMMON_UTILS_H_
#define MINDSPORE_MODELS_COMMON_UTILS_H_

#include <dirent.h>
#include <sys/stat.h>
#include <memory>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include "include/api/types.h"

bool GetDirFiles(const std::string &dir_name, std::vector<std::string> *sub_dirs, std::vector<std::string> *sub_files,
                 const std::vector<std::string> &extensions = {});
std::vector<std::string> GetAllFiles(const std::string &dirName, const std::vector<std::string> &extensions = {});
std::vector<std::vector<std::string>> GetAllInputData(const std::string &dir_name,
                                                      const std::vector<std::string> &extensions = {});

std::string RealPath(const std::string &path);
mindspore::MSTensor ReadFileToTensor(const std::string &file);

std::vector<std::string> GetAllFiles(const std::string &dirName, const std::vector<std::string> &extensions) {
  std::vector<std::string> dirs;
  std::vector<std::string> files;
  if (!GetDirFiles(dirName, &dirs, &files, extensions)) {
    return {};
  }
  if (dirs.empty()) {
    return files;
  }
  files.clear();
  for (auto &dir : dirs) {
    if (!GetDirFiles(dir, nullptr, &files, extensions)) {
      return {};
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::vector<std::vector<std::string>> GetAllInputData(const std::string &dir_name,
                                                      const std::vector<std::string> &extensions) {
  std::vector<std::vector<std::string>> ret;
  std::vector<std::string> sub_dirs;
  if (!GetDirFiles(dir_name, &sub_dirs, nullptr)) {
    return {};
  }
  std::sort(sub_dirs.begin(), sub_dirs.end());

  (void)std::transform(sub_dirs.begin(), sub_dirs.end(), std::back_inserter(ret),
                       [&extensions](const std::string &d) { return GetAllFiles(d, extensions); });

  return ret;
}

bool GetDirFiles(const std::string &dir_name, std::vector<std::string> *sub_dirs, std::vector<std::string> *sub_files,
                 const std::vector<std::string> &extensions) {
  if (dir_name.empty()) {
    std::cout << " dir_name is null ! " << std::endl;
    return false;
  }
  std::string real_path = RealPath(dir_name);
  struct stat s;
  lstat(real_path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dir_name is not a valid directory !" << std::endl;
    return false;
  }
  auto dir = opendir(real_path.c_str());
  if (dir == nullptr) {
    std::cout << "Can not open dir " << dir_name << std::endl;
    return false;
  }
  struct dirent *filename;
  while ((filename = readdir(dir)) != nullptr) {
    std::string d_name = std::string(filename->d_name);
    if (d_name == "." || d_name == "..") {
      continue;
    } else if (filename->d_type == DT_DIR) {
      if (sub_dirs) {
        sub_dirs->emplace_back(dir_name + "/" + d_name);
      }
    } else if (filename->d_type == DT_REG) {
      if (sub_files) {
        bool file_ok = true;
        if (!extensions.empty()) {
          file_ok = std::any_of(extensions.begin(), extensions.end(), [&d_name](const std::string &ext) {
            auto pos = d_name.find(ext);
            return pos != std::string::npos && pos + ext.size() == d_name.size();
          });
        }
        if (file_ok) {
          sub_files->emplace_back(dir_name + "/" + d_name);
        }
      }
    } else {
      continue;
    }
  }
  closedir(dir);
  return true;
}

std::vector<std::string> GetImagesById(const std::string &idFile, const std::string &dirName) {
  std::ifstream readFile(idFile);
  std::string line;
  std::vector<std::string> result;

  if (!readFile.is_open()) {
    std::cout << "can not open image id txt file" << std::endl;
    return result;
  }

  while (getline(readFile, line)) {
    std::size_t pos = line.find(" ");
    std::string id = line.substr(0, pos);
    result.emplace_back(dirName + "/" + id);
  }

  return result;
}

std::string RealPath(const std::string &path) {
  char realPathMem[PATH_MAX] = {0};
  char *realPathRet = nullptr;
  realPathRet = realpath(path.data(), realPathMem);
  if (realPathRet == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }
  std::string realPath(realPathMem);
  return realPath;
}

int WriteResult(const std::string &imageFile, const std::vector<mindspore::MSTensor> &outputs,
                const std::string &homePath = "./result_Files") {
  const int INVALID_POINTER = -1;
  const int ERROR = -2;
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::shared_ptr<const void> netOutput = outputs[i].Data();
    size_t outputSize = outputs[i].DataSize();
    int pos = imageFile.rfind('/');
    std::string fileName(imageFile, pos + 1);
    fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), '_' + std::to_string(i) + ".bin");
    std::string outFileName = homePath + "/" + fileName;
    FILE *outputFile = fopen(outFileName.c_str(), "wb");
    if (outputFile == nullptr) {
      std::cout << "open result file " << outFileName << " failed" << std::endl;
      return INVALID_POINTER;
    }
    size_t size = fwrite(netOutput.get(), sizeof(char), outputSize, outputFile);
    if (size != outputSize) {
      fclose(outputFile);
      std::cout << "write result file " << outFileName << " failed, write size[" << size
                << "] is smaller than output size[" << outputSize << "], maybe the disk is full." << std::endl;
      return ERROR;
    }
    fclose(outputFile);
  }
  return 0;
}

int WriteResultNoIndex(const std::string &imageFile, const std::vector<mindspore::MSTensor> &outputs,
                       const std::string &homePath = "./result_Files") {
  const int INVALID_POINTER = -1;
  const int ERROR = -2;
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::shared_ptr<const void> netOutput = outputs[i].Data();
    size_t outputSize = outputs[i].DataSize();
    int pos = imageFile.rfind('/');
    std::string fileName(imageFile, pos + 1);
    fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), ".bin");
    std::string outFileName = homePath + "/" + fileName;
    FILE *outputFile = fopen(outFileName.c_str(), "wb");
    if (outputFile == nullptr) {
      std::cout << "open result file " << outFileName << " failed" << std::endl;
      return INVALID_POINTER;
    }
    size_t size = fwrite(netOutput.get(), sizeof(char), outputSize, outputFile);
    if (size != outputSize) {
      fclose(outputFile);
      outputFile = nullptr;
      std::cout << "write result file " << outFileName << " failed, write size[" << size
                << "] is smaller than output size[" << outputSize << "], maybe the disk is full." << std::endl;
      return ERROR;
    }
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

DIR *OpenDir(const std::string &dirName) {
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
#endif  // MINDSPORE_MODELS_COMMON_UTILS_H_

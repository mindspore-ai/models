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

#include <fstream>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include "../inc/utils.h"

using mindspore::MSTensor;
using mindspore::DataType;

std::vector<std::string> GetAllFiles(std::string dirName) {
    std::ifstream txt_file(dirName);
    std::string line;
    std::vector<std::string> inputs_path;
    if (txt_file) {
        while (getline(txt_file, line)) {
            inputs_path.emplace_back(line);
        }
    } else {
        std::cout << "The file " << dirName << " is not exist" << std::endl;
        return {};
    }

  return inputs_path;
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

int WriteResult(const std::string& imageFile, const std::vector<MSTensor> &outputs) {
  std::string homePath = "./result_Files/";
  const int INVALID_POINTER = -1;
  const int ERROR = -2;
  for (size_t i = 0; i < outputs.size(); ++i) {
    size_t outputSize;
    std::shared_ptr<const void> netOutput;
    netOutput = outputs[i].Data();
    outputSize = outputs[i].DataSize();
    int pos = imageFile.rfind('/');
    std::string fileName(imageFile, pos + 1);
    std::string outFileName = homePath + fileName;
    std::string outs_file = "./result_Files/outputs.txt";
    FILE * outs_txt = fopen(outs_file.c_str(), "ab+");
    fprintf(outs_txt, "%s\n", outFileName.c_str());
    fclose(outs_txt);
    FILE * outputFile = fopen(outFileName.c_str(), "wb");
    if (outputFile == nullptr) {
        std::cout << "open result file " << outFileName << " failed" << std::endl;
        return INVALID_POINTER;
    }
    size_t size = fwrite(netOutput.get(), sizeof(char), outputSize, outputFile);
    if (size != outputSize) {
        fclose(outputFile);
        outputFile = nullptr;
        std::cout << "write result file " << outFileName << " failed, write size[" << size <<
            "] is smaller than output size[" << outputSize << "], maybe the disk is full." << std::endl;
        return ERROR;
    }
    fclose(outputFile);
    outputFile = nullptr;
  }
  return 0;
}

MSTensor ReadFileToTensor(const std::string &file) {
  if (file.empty()) {
  std::cout << "Pointer file is nullptr" << std::endl;
  return MSTensor();
  }

  std::ifstream ifs(file);
  if (!ifs.good()) {
  std::cout << "File: " << file << " is not exist" << std::endl;
  return MSTensor();
  }

  if (!ifs.is_open()) {
  std::cout << "File: " << file << "open failed" << std::endl;
  return MSTensor();
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  MSTensor buffer(file, mindspore::DataType::kNumberTypeFloat32, {static_cast<int64_t>(size)}, nullptr, size);

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
  DIR *dir;
  dir = opendir(realPath.c_str());
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

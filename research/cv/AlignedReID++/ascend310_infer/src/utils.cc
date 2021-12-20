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
#include <string>
#include <vector>
using mindspore::MSTensor;
using mindspore::DataType;

std::vector<std::string> split(const std::string &s, const std::string &separator) {
    std::vector<std::string> result;
    std::size_t i = 0;

    while (i != s.size()) {
        int flag = 0;
        while (i != s.size() && flag == 0) {
            flag = 1;
            for (std::size_t x = 0; x < separator.size(); ++x) {
                if (s[i] == separator[x]) {
                    ++i;
                    flag = 0;
                    break;
                }
            }
        }
        flag = 0;
        std::size_t j = i;
        while (j != s.size() && flag == 0) {
            for (std::size_t x = 0; x < separator.size(); ++x) {
                if (s[j] == separator[x]) {
                    flag = 1;
                    break;
                }
            }
            if (flag == 0) {
                ++j;
            }
        }
        if (i != j) {
            result.push_back(s.substr(i, j-i));
            i = j;
        }
    }
    return result;
}

std::vector<std::string> GetAllFiles(std::string_view dirName, const std::string& inputtype) {
    struct dirent *filename;
    DIR *dir = OpenDir(dirName);
    if (dir == nullptr) {
        return {};
    }
    std::vector<std::string> res;
    std::string homePath = "./" + inputtype + "_result_Files";
    std::string pidfilename = homePath + "/savepid.txt";
    std::string camidfilename = homePath + "/savecamid.txt";
    std::ofstream outfilepid;
    std::ofstream outfilecmd;
    outfilepid.open(pidfilename, std::ios::app);
    outfilecmd.open(camidfilename, std::ios::app);
    std::string pid;
    std::string camid;

    while ((filename = readdir(dir)) != nullptr) {
        std::string dName = std::string(filename->d_name);
        if (dName == "." || dName == ".." || filename->d_type != DT_REG) {
            continue;
        }
        if (dName == "Thumbs.db") {
            continue;
        }
        std::vector<std::string> v = split(dName, "_");
        pid = v[0];
        camid = v[1].substr(1, 1);
        if (pid == "-1") {
            continue;
        }
        outfilepid << pid << std::endl;
        outfilecmd << camid << std::endl;
        res.emplace_back(std::string(dirName) + "/" + filename->d_name);
    }
    for (auto &f : res) {
        std::cout << "image file: " << f << std::endl;
    }
    return res;
}

int WriteResult(const std::string& imageFile, const std::vector<MSTensor> &outputs) {
    std::string homePath = "./result_Files";
    const int INVALID_POINTER = -1;
    const int ERROR = -2;
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        std::shared_ptr<const void> netOutput = outputs[i].Data();
        outputSize = outputs[i].DataSize();

        int pos = imageFile.rfind('/');
        std::string fileName(imageFile, pos + 1);
        fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), '_' + std::to_string(i) + ".txt");
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
            std::cout << "write result file " << outFileName << " failed, write size[" << size <<
                "] is smaller than output size[" << outputSize << "], maybe the disk is full." << std::endl;
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

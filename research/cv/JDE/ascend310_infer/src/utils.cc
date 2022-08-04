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

/* This file was copied from project [whu_mmap_cliang][Course_NNDL] */

#include "inc/utils.h"
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include <iostream>
using mindspore::DataType;
using mindspore::MSTensor;

std::string file_dirs_2_file_bin(std::string imageFile, const std::string &dataset_path) {
    std::string homePath = "./result_Files";
    int data_path_pos = dataset_path.length();
    std::string fileName(imageFile, data_path_pos + 1);
    int pos = fileName.find('/');
    while (pos> 0) {
        fileName = fileName.replace(pos, 1, "_");
        pos = fileName.find('/');
    }
    fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), ".bin");
    std::string outFileName = homePath + "/" + fileName;
    return outFileName;
}

std::vector<std::string> GetAllFiles(std::string dirName1, const std::string &endswith = ".jpg") {
    struct dirent *filename;
    std::vector<std::string> dirs;
    std::vector<std::string> files;
    dirs.emplace_back(dirName1);
    while (dirs.size()) {
        std::string dirName = dirs.back();
        dirs.pop_back();
        DIR *dir = OpenDir(dirName);
        if (dir == nullptr) continue;
        while ((filename = readdir(dir)) != nullptr) {
            std::string dName = std::string(filename->d_name);
            if (dName == "." || dName == "..") {
                continue;
            }
            if (filename->d_type == DT_DIR) {
                dirs.emplace_back(std::string(dirName) + "/" + filename->d_name);
            }
            if (filename->d_type == DT_REG) {
                std::string dName2 = std::string(filename->d_name);
                if (dName2 == "." || dName2 == ".." || filename->d_type != DT_REG) {
                    continue;
                }
                if (dName2.substr(dName2.length() - endswith.length(), endswith.length()) != endswith) {
                    continue;
                }
                std::string imageFile = std::string(dirName) + "/" + filename->d_name;
                files.emplace_back(imageFile);
            }
        }
    }
    std::sort(files.begin(), files.end());
    std::cout << "image file num: " << files.size() << std::endl;
    return files;
}

int WriteResult_old(const std::string &imageFile, const std::vector<MSTensor> &outputs) {
    std::string homePath = "./result_Files";
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        std::shared_ptr<const void> netOutput;
        netOutput = outputs[i].Data();
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

int WriteResult(const std::string &dataset_path, const std::string &imageFile, const std::vector<MSTensor> &outputs) {
    auto tensor = outputs[1];
    std::string outFileName = file_dirs_2_file_bin(imageFile, dataset_path);
    return WriteTensorToFile(outFileName, tensor);
}

int WriteTensorToFile(const std::string &file, mindspore::MSTensor tensor) {
    FILE *outputFile = fopen(file.c_str(), "wb");
    size_t outputSize;
    std::shared_ptr<const void> netOutput;
    netOutput = tensor.Data();
    outputSize = tensor.DataSize();
    fwrite(netOutput.get(), outputSize, sizeof(char), outputFile);
    fclose(outputFile);
    outputFile = nullptr;
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
    return realPath;
}

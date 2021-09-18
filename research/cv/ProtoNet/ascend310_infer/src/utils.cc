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
#include "../inc/utils.h"

namespace ms = mindspore;
using mindspore::MSTensor;
using std::vector;
using std::string;
using std::string_view;
using std::sort;
using std::shared_ptr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;

vector<string> GetAllFiles(string_view dir_name) {
  struct dirent *filename;
  DIR *dir = OpenDir(dir_name);
  if (dir == nullptr) {
    return {};
  }

  /* read all the files in the dir ~ */
  vector<string> res;
  while ((filename = readdir(dir)) != nullptr) {
    string d_name = string(filename->d_name);
    // get rid of "." and ".."
    if (d_name == "." || d_name == ".." || filename->d_type != DT_REG)
      continue;
    res.emplace_back(string(dir_name) + "/" + filename->d_name);
  }

  sort(res.begin(), res.end());
  return res;
}


int WriteResult(const string& imageFile, const vector<MSTensor> &outputs) {
    string homePath = "./result_Files/";
    const int INVALID_POINTER = -1;
    const int ERROR = -2;
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        shared_ptr<const void> netOutput = outputs[i].Data();
        outputSize = outputs[i].DataSize();
        int pos = imageFile.rfind('/');
        string fileName(imageFile, pos + 1);
        string outFileName = homePath + "/" + fileName;
        FILE *outputFile = fopen(outFileName.c_str(), "wb");
        if (outputFile == nullptr) {
            cout << "open result file " << outFileName << " failed" << endl;
            return INVALID_POINTER;
        }
        size_t size = fwrite(netOutput.get(), sizeof(char), outputSize, outputFile);
        if (size != outputSize) {
            fclose(outputFile);
            outputFile = nullptr;
            cout << "write result file " << outFileName << " failed, write size[" << size <<
                "] is smaller than output size[" << outputSize << "], maybe the disk is full." << endl;
            return ERROR;
        }
        fclose(outputFile);
        outputFile = nullptr;
    }
    return 0;
}





DIR *OpenDir(string_view dir_name) {
  // check the parameter !
  if (dir_name.empty()) {
    cout << " dir_name is null ! " << endl;
    return nullptr;
  }

  string real_path = RealPath(dir_name);

  // check if dir_name is a valid dir
  struct stat s;
  lstat(real_path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    cout << "dir_name is not a valid directory !" << endl;
    return nullptr;
  }

  DIR *dir;
  dir = opendir(real_path.c_str());
  if (dir == nullptr) {
    cout << "Can not open dir " << dir_name << endl;
    return nullptr;
  }
  return dir;
}



string RealPath(string_view path) {
  char real_path_mem[PATH_MAX] = {0};
  char *real_path_ret = realpath(path.data(), real_path_mem);

  if (real_path_ret == nullptr) {
    cout << "File: " << path << " is not exist.";
    return "";
  }

  return string(real_path_mem);
}



ms::MSTensor ReadFile(const string &file) {
  if (file.empty()) {
    cout << "Pointer file is nullptr" << endl;
    return ms::MSTensor();
  }

  ifstream ifs(file);
  if (!ifs.good()) {
    cout << "File: " << file << " is not exist" << endl;
    return ms::MSTensor();
  }

  if (!ifs.is_open()) {
    cout << "File: " << file << "open failed" << endl;
    return ms::MSTensor();
  }

  ifs.seekg(0, ios::end);
  size_t size = ifs.tellg();
  ms::MSTensor buffer(file, ms::DataType::kNumberTypeUInt8, {static_cast<int64_t>(size)}, nullptr, size);

  ifs.seekg(0, ios::beg);
  ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
  ifs.close();

  return buffer;
}
size_t GetMax(ms::MSTensor data) {
  float max_value = -1;
  size_t max_idx = 0;
  const float *p = reinterpret_cast<const float *>(data.MutableData());
  for (size_t i = 0; i < data.DataSize() / sizeof(float); ++i) {
    if (p[i] > max_value) {
      max_value = p[i];
      max_idx = i;
    }
  }
  return max_idx;
}

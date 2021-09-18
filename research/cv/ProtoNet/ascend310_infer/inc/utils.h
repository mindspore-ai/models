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

#ifndef MINDSPORE_INFERENCE_UTILS_H_
#define MINDSPORE_INFERENCE_UTILS_H_

#include <sys/stat.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <memory>
#include "include/api/types.h"

namespace ms = mindspore;
// using namespace std;
using std::vector;
using std::string;
using std::string_view;


vector<string> GetAllFiles(string_view dir_name);
DIR *OpenDir(string_view dir_name);
string RealPath(string_view path);
ms::MSTensor ReadFile(const string &file);
size_t GetMax(ms::MSTensor data);
int WriteResult(const string& imageFile, const vector<mindspore::MSTensor> &outputs);

#endif

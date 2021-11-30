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

#include <sys/stat.h>
#include <sys/time.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"

namespace ms = mindspore;
namespace ds = mindspore::dataset;

int WriteResult(const std::string& imageFile, const std::vector<ms::MSTensor> &outputs);
void print_shape(const ms::MSTensor &buffer);
std::vector<std::string> GetAllFiles(std::string_view dir_name);
DIR *OpenDir(std::string_view dir_name);
std::string RealPath(std::string_view path);
ms::MSTensor ReadFile(const std::string &file);

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

#include <dirent.h>
#include <sys/stat.h>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include "include/api/types.h"

float WriteResult(const std::string &data_dir, const std::vector<mindspore::MSTensor> &outputs,
                  const std::vector<uint32_t> &gt, int32_t points_num);
std::string print_data_shape(const std::vector<int64_t> &shape);
void GetDataSample(const std::string &data_dir, int32_t &points_num, std::vector<float> &input,
                   std::vector<uint32_t> &gt);
std::vector<std::string> GetDataDirs(const std::string &datasets_dir, const std::string &precision);
std::string print_dataptr(const void *data_ptr, size_t size);

#endif

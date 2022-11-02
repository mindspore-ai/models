// Copyright 2022 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include "./ms_ext.h"
#include <string.h>
#include <torch/extension.h>

#include <vector>
#include <string>
#include <unordered_map>

int8_t GetDtype(const std::string& dtypes) {
  int8_t type = 6;
  std::unordered_map<std::string, int8_t> m{
      {"uint8", 0}, {"int8", 1},    {"int16", 2},   {"int32", 3},
      {"int64", 4}, {"float16", 5}, {"float32", 6}, {"float64", 7}};
  if (m.count(dtypes)) {
    type = m[dtypes];
  }
  return type;
}

std::vector<at::Tensor> get_torch_tensors(int nparam, void** params, int* ndims,
                                          int64_t** shapes, const char** dtypes,
                                          c10::Device device) {
  std::vector<at::Tensor> tensors;
  for (int i = 0; i < nparam; i++) {
    std::vector<int64_t> size;
    for (int j = 0; j < ndims[i]; j++) {
      size.push_back(shapes[i][j]);
    }
    int8_t type = GetDtype(dtypes[i]);
    auto option = at::TensorOptions()
                      .dtype(static_cast<c10::ScalarType>(type))
                      .device(device);
    tensors.emplace_back(at::from_blob(params[i], size, option));
  }
  return tensors;
}

void output_memcpy(void* output, const torch::Tensor& t) {
  memcpy(output, t.data_ptr(), t.element_size() * t.numel());
}

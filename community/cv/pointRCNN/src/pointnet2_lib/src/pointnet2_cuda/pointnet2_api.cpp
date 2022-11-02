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

#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "./group_points_gpu.h"
#include "./interpolate_gpu.h"
#include "./ball_query_gpu.h"
#include "./sampling_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query_wrapper", &ball_query_wrapper_fast,
        "ball_query_wrapper_fast");

  m.def("group_points_wrapper", &group_points_wrapper_fast,
        "group_points_wrapper_fast");
  m.def("group_points_grad_wrapper", &group_points_grad_wrapper_fast,
        "group_points_grad_wrapper_fast");

  m.def("gather_points_wrapper", &gather_points_wrapper_fast,
        "gather_points_wrapper_fast");
  m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper_fast,
        "gather_points_grad_wrapper_fast");

  m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper,
        "furthest_point_sampling_wrapper");

  m.def("three_nn_wrapper", &three_nn_wrapper_fast, "three_nn_wrapper_fast");
  m.def("three_interpolate_wrapper", &three_interpolate_wrapper_fast,
        "three_interpolate_wrapper_fast");
  m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper_fast,
        "three_interpolate_grad_wrapper_fast");
}

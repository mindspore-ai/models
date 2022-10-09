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

#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

#include <vector>

int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample,
                            at::Tensor new_xyz_tensor, at::Tensor xyz_tensor,
                            at::Tensor idx_tensor);

void ball_query_kernel_launcher_fast(int b, int n, int m, float radius,
                                     int nsample, const float *xyz,
                                     const float *new_xyz, int *idx,
                                     cudaStream_t stream);

#endif

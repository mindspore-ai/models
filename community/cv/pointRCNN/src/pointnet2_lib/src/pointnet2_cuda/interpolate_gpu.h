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

#ifndef _INTERPOLATE_GPU_H
#define _INTERPOLATE_GPU_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

#include <vector>

void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor,
                           at::Tensor known_tensor, at::Tensor dist2_tensor,
                           at::Tensor idx_tensor);

void three_nn_kernel_launcher_fast(int b, int n, int m, const float *unknown,
                                   const float *known, float *dist2, int *idx,
                                   cudaStream_t stream);

void three_interpolate_wrapper_fast(int b, int c, int m, int n,
                                    at::Tensor points_tensor,
                                    at::Tensor idx_tensor,
                                    at::Tensor weight_tensor,
                                    at::Tensor out_tensor);

void three_interpolate_kernel_launcher_fast(int b, int c, int m, int n,
                                            const float *points, const int *idx,
                                            const float *weight, float *out,
                                            cudaStream_t stream);

void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m,
                                         at::Tensor grad_out_tensor,
                                         at::Tensor idx_tensor,
                                         at::Tensor weight_tensor,
                                         at::Tensor grad_points_tensor);

void three_interpolate_grad_kernel_launcher_fast(
    int b, int c, int n, int m, const float *grad_out, const int *idx,
    const float *weight, float *grad_points, cudaStream_t stream);

#endif

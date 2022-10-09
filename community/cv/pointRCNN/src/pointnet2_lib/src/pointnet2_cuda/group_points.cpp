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

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>

#include <vector>

#include "./group_points_gpu.h"
#include "./ms_ext.h"

int group_points_grad_wrapper_fast(int b, int c, int n, int npoints,
                                   int nsample, at::Tensor grad_out_tensor,
                                   at::Tensor idx_tensor,
                                   at::Tensor grad_points_tensor) {
  float* grad_points = grad_points_tensor.data_ptr<float>();
  const int* idx = idx_tensor.data_ptr<int>();
  const float* grad_out = grad_out_tensor.data_ptr<float>();

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  group_points_grad_kernel_launcher_fast(b, c, n, npoints, nsample, grad_out,
                                         idx, grad_points, stream);
  return 1;
}

extern "C" int ms_group_points_grad_wrapper_fast(int nparam, void** params,
                                                 int* ndims, int64_t** shapes,
                                                 const char** dtypes,
                                                 void* stream, void* extra) {
  auto tensors =
      get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
  auto b = tensors[0].item<int>();
  auto c = tensors[1].item<int>();
  auto n = tensors[2].item<int>();
  auto npoints = tensors[3].item<int>();
  auto nsample = tensors[4].item<int>();
  group_points_grad_wrapper_fast(b, c, n, npoints, nsample, tensors[5],
                                 tensors[6], tensors[7]);
  return 0;
}

int group_points_wrapper_fast(int b, int c, int n, int npoints, int nsample,
                              at::Tensor points_tensor, at::Tensor idx_tensor,
                              at::Tensor out_tensor) {
  const float* points = points_tensor.data_ptr<float>();
  const int* idx = idx_tensor.data_ptr<int>();
  float* out = out_tensor.data_ptr<float>();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  group_points_kernel_launcher_fast(b, c, n, npoints, nsample, points, idx, out,
                                    stream);
  return 1;
}

extern "C" int ms_group_points_wrapper_fast(int nparam, void** params,
                                            int* ndims, int64_t** shapes,
                                            const char** dtypes, void* stream,
                                            void* extra) {
  auto tensors =
      get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
  auto b = tensors[0].item<int>();
  auto c = tensors[1].item<int>();
  auto n = tensors[2].item<int>();
  auto npoints = tensors[3].item<int>();
  auto nsample = tensors[4].item<int>();
  group_points_wrapper_fast(b, c, n, npoints, nsample, tensors[5], tensors[6],
                            tensors[7]);
  return 0;
}

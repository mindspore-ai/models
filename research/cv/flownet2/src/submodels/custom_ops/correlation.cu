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

#include <thrust/reduce.h>
#include <stdio.h>
#include <algorithm>

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32
#define FULL_MASK 0xffffffff

__global__ void correlationInitKernel(size_t size_init, float *input) {
    auto idx = blockIdx.x * CUDA_NUM_THREADS + threadIdx.x;
    if (idx < size_init) {
        input[idx] = static_cast<float>(.0);
    }
}


__forceinline__ __device__ float warpReduceSum(float value) {
        for (int offset = 16; offset > 0; offset /= 2)
                value += __shfl_down_sync(FULL_MASK, value, offset);
        return value;
}


__forceinline__ __device__ float blockReduceSum(float value) {
        static __shared__ float shared[32];
        int lane = threadIdx.x % warpSize;
        int windex = threadIdx.x / warpSize;
        value = warpReduceSum(value);
        if (lane == 0)
                shared[windex] = value;

        __syncthreads();

        value = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

        if (windex == 0)
                value = warpReduceSum(value);
        return value;
}

__global__ void correlation_forward(float*  output,
                const int tdim_cyx, const int tdim_yx, const int tdim_x,
                const float* padded_input1, const float* padded_input2,
                const int pdim_yxc, const int pdim_xc, const int pdim_c,
                const int kernel_size, const int max_displacement, const int stride1, const int stride2) {
        int32_t kernel_radius = (kernel_size - 1) / 2;
        int32_t displacement_radius = max_displacement / stride2;
        int32_t displacement_size = 2 * displacement_radius + 1;

        int32_t nums = kernel_size * kernel_size * pdim_c;

        int32_t n = blockIdx.x;
        int32_t y1 = blockIdx.y * stride1 + max_displacement;
        int32_t x1 = blockIdx.z * stride1 + max_displacement;
        int32_t c = threadIdx.x;

        // along channel axism, do element-wise product
        for (int t_j = -displacement_radius; t_j <= displacement_radius; ++t_j) {
                for (int t_i = -displacement_radius; t_i <= displacement_radius; ++t_i) {
                        int x2 = x1 + t_i * stride2;
                        int y2 = y1 + t_j * stride2;
                        float acc = 0.0f;
                        // add 2 feature kernel_radius
                        for (int j = -kernel_radius; j <= kernel_radius; ++j) {
                                for (int i = -kernel_radius; i <= kernel_radius; ++i) {
                                        #pragma unroll
                                        for (int ch = c; ch < pdim_c; ch += blockDim.x) {
                                                int index1 = n * pdim_yxc + (y1 + j) * pdim_xc + (x1 + i) * pdim_c + ch;
                                                int index2 = n * pdim_yxc + (y2 + j) * pdim_xc + (x2 + i) * pdim_c + ch;
                                                acc += static_cast<float>(padded_input1[index1] *
                                                                          padded_input2[index2]);
                                        }
                                }
                        }

                        if (blockDim.x == warpSize) {
                            __syncwarp();
                            acc = warpReduceSum(acc);
                        } else {
                            __syncthreads();
                            acc = blockReduceSum(acc);
                        }

                        if (threadIdx.x == 0) {
                                int tc = (t_j + displacement_radius) * displacement_size
                                                + (t_i + displacement_radius);
                                const int tindex = n * tdim_cyx + tc * tdim_yx + blockIdx.y * tdim_x + blockIdx.z;
                                output[tindex] = static_cast<float>(acc / nums);
                        }
                }
        }
}

extern "C" int correlation(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  constexpr int OUTPUT_INDEX = 2;
  constexpr int INPUT_INDEX = 0;
  constexpr int TOTAL_PARAM_NUM = 3;
  if (nparam != TOTAL_PARAM_NUM) {
    return 1;
  }
  // This is to check if the type of parameters the same as what the user wants.
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }
  // input1's index is 0, input2's index is 1 and output's index is 2
  float *input1 = static_cast<float *>(params[0]);
  float *input2 = static_cast<float *>(params[1]);
  float *output = static_cast<float *>(params[2]);

  int batchSize = shapes[OUTPUT_INDEX][0];
  int outputChannels = shapes[OUTPUT_INDEX][1];
  int outputHeight = shapes[OUTPUT_INDEX][2];
  int outputWidth = shapes[OUTPUT_INDEX][3];
  int inputChannels = shapes[INPUT_INDEX][3];
  int inputHeight = shapes[INPUT_INDEX][1];
  int inputWidth = shapes[INPUT_INDEX][2];

  // notice: At Currently the parameter used in cuda is fixed because the interface have no place to pass parameters
  // need to be changed in future
  const int kernel_size = 1;
  const int max_displacement = 20;
  const int stride1 = 1;
  const int stride2 = 2;

  int output_size = batchSize*outputChannels*outputWidth*outputHeight;
  int n = output_size / CUDA_NUM_THREADS;
  correlationInitKernel<<<n + 1, CUDA_NUM_THREADS, 0, custream>>>(output_size, output);

  dim3 threadsPerBlock(THREADS_PER_BLOCK);
  dim3 totalBlocksCorr(batchSize, outputHeight, outputWidth);

  int32_t pdim_yxc = inputHeight * inputWidth * inputChannels;
  int32_t pdim_xc = inputWidth * inputChannels;
  int32_t pdim_c = inputChannels;

  int32_t tdim_cyx = outputChannels * outputHeight * outputWidth;
  int32_t tdim_yx = outputHeight * outputWidth;
  int32_t tdim_x = outputWidth;

  correlation_forward<<<totalBlocksCorr, threadsPerBlock, 0, custream>>>
                        (output, tdim_cyx, tdim_yx, tdim_x,
                         input1, input2, pdim_yxc, pdim_xc, pdim_c,
                         kernel_size, max_displacement, stride1, stride2);
  return 0;
}

// correlation_backward_input1 kernel
__global__ void correlation_backward_input1(int item, float *grad_input_1,
                                            const int p_dim_yxc, const int p_dim_xc, const int p_dim_c,
                                            const int o_dim_cyx, const int o_dim_yx, const int o_dim_x,
                                            const float *gradOutput, int outputChannels,
                                            int outputHeight, int outputWidth,
                                            const float *padded_input2, int pad_size,
                                            int kernel_size, int max_displacement,
                                            int stride1, int stride2, int kernel_radius, int displacement_radius,
                                            int displacement_size) {
    // NCHW (bs,num of channels,height,width)
    int n = item;
    int y = blockIdx.x * stride1 + pad_size;
    int x = blockIdx.y * stride1 + pad_size;
    int c = blockIdx.z;
    int tch_off = threadIdx.x;

    int t_dim_cyx = outputChannels * outputHeight * outputWidth;
    int t_dim_yx = outputHeight * outputWidth;
    int t_dim_x = outputWidth;

    int x_min = (x - kernel_radius - max_displacement) / stride1;
    int y_min = (y - kernel_radius - max_displacement) / stride1;
    int x_max = (x + kernel_radius - max_displacement) / stride1;
    int y_max = (y + kernel_radius - max_displacement) / stride1;

    // grad_input_1 is zero filled
    if (x_max < 0 || y_max < 0 || x_min >= outputWidth || y_min >= outputHeight
        || x_min > x_max || y_min > y_max) {
        return;
    }
    // add range limit of height and width to cal grad_input_1
    x_min = max(0, x_min);
    x_max = min(outputWidth-1, x_max);

    y_min = max(0, y_min);
    y_max = min(outputHeight-1, y_max);

    float nums = kernel_size * kernel_size * p_dim_c;

    __shared__ float temp_sum[THREADS_PER_BLOCK];
    temp_sum[tch_off] = 0;
    // along channel axism
    for (int tc = tch_off; tc < outputChannels; tc += THREADS_PER_BLOCK) {
      int m_2 = (tc % displacement_size - displacement_radius) * stride2;
      int n_2 = (tc / displacement_size - displacement_radius) * stride2;
      int index2 =  n * p_dim_yxc + (y + n_2) * p_dim_xc + (x + m_2) * p_dim_c + c;

      float val2 = padded_input2[index2];

      for (int j = y_min; j <= y_max; ++j) {
        for (int i = x_min; i <= x_max; ++i) {
          int t_index = n * t_dim_cyx + tc * t_dim_yx + j * t_dim_x + i;
          temp_sum[tch_off] += gradOutput[t_index] * val2;
        }
      }
    }
    __syncthreads();

    if (tch_off == 0) {
      float reduce_sum = 0;
      for (int index = 0; index < THREADS_PER_BLOCK; index++) {
          reduce_sum += temp_sum[index];
      }
      const int index1 = n * o_dim_cyx + c * o_dim_yx + (y - pad_size) * o_dim_x + (x - pad_size);
      grad_input_1[index1] = reduce_sum / nums;
    }
}

// correlation_backward_input2 kernel
__global__ void correlation_backward_input2(int item, float  *grad_input_2,
                                            const int p_dim_yxc, const int p_dim_xc, const int p_dim_c,
                                            const int o_dim_cyx, const int o_dim_yx, const int o_dim_x,
                                            const int t_dim_cyx, const int t_dim_yx, const int t_dim_x,
                                            const float *gradOutput, int outputChannels,
                                            int outputHeight, int outputWidth,
                                            const float *padded_input1, int pad_size,
                                            int kernel_size, int max_displacement,
                                            int stride1, int stride2, int kernel_radius, int displacement_radius,
                                            int displacement_size) {
    // NCHW (bs,num of channels,height,width)
    int n = item;
    int y = blockIdx.x * stride1 + pad_size;
    int x = blockIdx.y * stride1 + pad_size;
    int c = blockIdx.z;

    int tch_off = threadIdx.x;
    __shared__ float prod_sum[THREADS_PER_BLOCK];
    prod_sum[tch_off] = 0;
    for (int tc = tch_off; tc < outputChannels; tc += THREADS_PER_BLOCK) {
      int m_1 = (tc % displacement_size - displacement_radius) * stride2;
      int n_1 = (tc / displacement_size - displacement_radius) * stride2;

      int x_min = (x - kernel_radius - max_displacement - m_1) / stride1;
      int y_min = (y - kernel_radius - max_displacement - n_1) / stride1;

      int x_max = (x + kernel_radius - max_displacement - m_1) / stride1;
      int y_max = (y + kernel_radius - max_displacement - n_1) / stride1;

      if (x_max < 0 || y_max < 0) {
          continue;
      }
      if (x_min >= outputWidth || y_min >= outputHeight) {
          continue;
      }
      if (x_min > x_max || y_min > y_max) {
          continue;
      }

      // add range limit of height and width to cal grad_input_2
      x_min = max(0, x_min);
      x_max = min(outputWidth-1, x_max);
      y_min = max(0, y_min);
      y_max = min(outputHeight-1, y_max);

      // assign value of gradOutput to grad_input_2
      int index_1 = n * p_dim_yxc + (y - n_1) * p_dim_xc + (x - m_1) * p_dim_c + c;
      float val_1 = padded_input1[index_1];
      for (int j = y_min; j <= y_max; ++j) {
         for (int i = x_min; i <= x_max; ++i) {
            int t_index = n * t_dim_cyx + tc * t_dim_yx + j * t_dim_x + i;
            prod_sum[tch_off] += gradOutput[t_index] * val_1;
         }
      }
    }

    __syncthreads();
    if (tch_off == 0) {
      float reduce_sum = 0;
      for (int index = 0; index < THREADS_PER_BLOCK; index++) {
          reduce_sum += prod_sum[index];
      }
      const int index_2 = n * o_dim_cyx + c * o_dim_yx + (y - pad_size) * o_dim_x + (x - pad_size);
      float nums = kernel_size * kernel_size * p_dim_c;
      grad_input_2[index_2] = reduce_sum / nums;
    }
}

extern "C" int correlationGrad(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                     void *stream, void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    constexpr int INPUT1_INDEX = 0;
    constexpr int GRAD_OUTPUT_INDEX = 2;
    constexpr int TOTAL_PARAM_NUM = 5;

    if (nparam != TOTAL_PARAM_NUM) {
         return 1;
    }
    // This is to check if the type of parameters the same as what the user wants.
    for (int i = 0; i < TOTAL_PARAM_NUM; i++) {
        if (strcmp(dtypes[0], "float32") != 0) {
            return 2;
        }
    }

    float *padded_input1 = static_cast<float *>(params[0]);
    float *padded_input2 = static_cast<float *>(params[1]);
    float *gradOutput = static_cast<float *>(params[2]);
    float *gradInput1 = static_cast<float *>(params[3]);
    float *gradInput2 = static_cast<float *>(params[4]);

    int batchSize = shapes[GRAD_OUTPUT_INDEX][0];
    int outputChannels = shapes[GRAD_OUTPUT_INDEX][1];
    int outputHeight = shapes[GRAD_OUTPUT_INDEX][2];
    int outputWidth = shapes[GRAD_OUTPUT_INDEX][3];

    int inputChannels = shapes[INPUT1_INDEX][3];
    int p_inputHeight = shapes[INPUT1_INDEX][1];
    int p_inputWidth = shapes[INPUT1_INDEX][2];

    // notice: At Currently the parameter used in cuda is fixed because the interface have no place to pass parameters
    // need to be changed in future
    const int pad_size = 20;
    const int kernel_size = 1;
    const int max_displacement = 20;
    const int stride1 = 1;
    const int stride2 = 2;

    int inputWidth = p_inputWidth - 2 * pad_size;
    int inputHeight = p_inputHeight - 2 * pad_size;

    int kernel_radius = (kernel_size - 1) / 2;
    int displacement_radius = max_displacement / stride2;
    int displacement_size = 2 * displacement_radius + 1;

    int p_dim_yxc = p_inputHeight * p_inputWidth * inputChannels;
    int p_dim_xc = p_inputWidth * inputChannels;
    int p_dim_c = inputChannels;

    int t_dim_cyx = outputChannels * outputHeight * outputWidth;
    int t_dim_yx = outputHeight * outputWidth;
    int t_dim_x = outputWidth;

    int o_dim_cyx = inputChannels * inputHeight* inputWidth;
    int o_dim_yx = inputHeight * inputWidth;
    int o_dim_x = inputWidth;

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 totalBlocksCorr(inputHeight, inputWidth, inputChannels);

    // initialize gradInput1 zero
    int gradInput1_size = batchSize*inputChannels*inputWidth*inputHeight;
    correlationInitKernel<<<gradInput1_size / CUDA_NUM_THREADS + 1, CUDA_NUM_THREADS,
                            0, custream>>>(gradInput1_size, gradInput1);
    // call correlation_backward_input1
    for (int n = 0; n < batchSize; ++n) {
        correlation_backward_input1<<<totalBlocksCorr, threadsPerBlock, 0, custream>>> (
              n, gradInput1, p_dim_yxc, p_dim_xc, p_dim_c, o_dim_cyx, o_dim_yx, o_dim_x,
              gradOutput, outputChannels, outputHeight, outputWidth,
              padded_input2, pad_size, kernel_size, max_displacement, stride1, stride2,
              kernel_radius, displacement_radius, displacement_size);
    }
    // initialize gradInput2 zero
    int gradInput2_size = batchSize*inputChannels*inputWidth*inputHeight;
    correlationInitKernel<<<gradInput2_size / CUDA_NUM_THREADS + 1, CUDA_NUM_THREADS,
                            0, custream>>>(gradInput2_size, gradInput2);
    // call correlation_backward_input2
    for (int n = 0; n < batchSize; n++) {
      correlation_backward_input2<<<totalBlocksCorr, threadsPerBlock, 0, custream>>>(
            n, gradInput2, p_dim_yxc, p_dim_xc, p_dim_c, o_dim_cyx, o_dim_yx, o_dim_x,
            t_dim_cyx, t_dim_yx, t_dim_x,
            gradOutput, outputChannels, outputHeight, outputWidth,
            padded_input1, pad_size, kernel_size, max_displacement, stride1, stride2,
            kernel_radius, displacement_radius, displacement_size);
    }
    return 0;
}

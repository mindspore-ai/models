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
#define CUDA_NUM_THREADS 512
#define THREADS_PER_BLOCK 64

#include <algorithm>

__device__ __forceinline__ float MsAtomicAdd(float *address, const float val) {
  return atomicAdd(address, val);
}

__global__ void Resample2dInitKernel(size_t size_init, float *input) {
  auto idx = blockIdx.x * CUDA_NUM_THREADS + threadIdx.x;
  if (idx < size_init) {
     input[idx] = static_cast<float>(.0);
  }
}


__device__ int GET_INDEX(const int batch , const int channels, const int height, const int width,
                        const int batch_stride , const int channels_stride, const int height_stride) {
     return batch*batch_stride+channels*channels_stride+height*height_stride+width;
}

__device__ float DIM3_INDEX(const float *input, const int batch , const int channels, const int height, const int width,
                        const int batch_stride , const int channels_stride, const int height_stride) {
    return input[batch*batch_stride+channels*channels_stride+height*height_stride+width];
}


__global__ void Resample2dKernel(size_t size, const float *input1, const float *input2, float *out_data,
                    int batch_stride_x1, int channel_stride_x1, int height_stride_x1,
                    int batch_stride_x2, int channel_stride_x2, int height_stride_x2,
                    int batch_output, int channel_output, int height_output, int width_output,
                    int kernel_size, bool bilinear) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    float val = 0.0;

    int dim_b = batch_output;
    int dim_c = channel_output;
    int dim_h = height_output;
    int dim_w = width_output;
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;
    int b = (index / dim_chw) % dim_b;
    int c = (index / dim_hw) % dim_c;
    int y = (index / dim_w) % dim_h;
    int x = (index) % dim_w;

    float dx = DIM3_INDEX(input2, b, 0, y, x, batch_stride_x2, channel_stride_x2, height_stride_x2);
    float dy = DIM3_INDEX(input2, b, 1, y, x, batch_stride_x2, channel_stride_x2, height_stride_x2);

    float xf = x + dx;
    float yf = y + dy;  // img+flow
    float alpha = xf - (floor(xf));  // alpha
    float beta = yf - (floor(yf));  // beta
    if (bilinear) {
        int xL = max(min(static_cast<int>(floor(xf)),    dim_w-1), 0);
        int xR = max(min(static_cast<int>(floor(xf)+1), dim_w -1), 0);
        int yT = max(min(static_cast<int>(floor(yf)),    dim_h-1), 0);
        int yB = max(min(static_cast<int>(floor(yf)+1),  dim_h-1), 0);
        for (int fy = 0; fy < kernel_size; fy += 1) {
            for (int fx = 0; fx < kernel_size; fx += 1) {
                float offTL = DIM3_INDEX(input1, b, c, yT + fy, xL + fx,
                                    batch_stride_x1, channel_stride_x1, height_stride_x1);
                float offTR = DIM3_INDEX(input1, b, c, yT + fy, xR + fx,
                                    batch_stride_x1, channel_stride_x1, height_stride_x1);
                float offBL = DIM3_INDEX(input1, b, c, yB + fy, xL + fx,
                                    batch_stride_x1, channel_stride_x1, height_stride_x1);
                float offBR = DIM3_INDEX(input1, b, c, yB + fy, xR + fx,
                                    batch_stride_x1, channel_stride_x1, height_stride_x1);
                val += (1. - alpha)*(1. - beta) *  offTL;
                val += (alpha)*(1. - beta) * offTR;
                val += (1. - alpha)*(beta) * offBL;
                val += (alpha)*(beta) * offBR;
            }
        }
        out_data[index] = val;
    } else {
        int xN = max(min(static_cast<int>(floor(xf + 0.5)), dim_w - 1), 0);
        int yN = max(min(static_cast<int>(floor(yf + 0.5)), dim_h - 1), 0);
        out_data[index] = DIM3_INDEX(input1, b, c, yN, xN, batch_stride_x1, channel_stride_x1, height_stride_x1);
    }
}



extern "C" int Resample2d(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    constexpr int INPUT1_INDEX = 0;
    constexpr int INPUT2_INDEX = 1;
    constexpr int OUTPUT_INDEX = 2;
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

    float *x1 = static_cast<float *>(params[0]);
    float *x2 = static_cast<float *>(params[1]);

    float *out_data = static_cast<float *>(params[2]);

//     int batch_x1 = shapes[INPUT1_INDEX][0];
    int channel_x1 = shapes[INPUT1_INDEX][1];
    int height_x1 = shapes[INPUT1_INDEX][2];
    int width_x1 = shapes[INPUT1_INDEX][3];

//     int batch_x2 = shapes[INPUT2_INDEX][0];
    int channel_x2 = shapes[INPUT2_INDEX][1];
    int height_x2 = shapes[INPUT2_INDEX][2];
    int width_x2 = shapes[INPUT2_INDEX][3];

    int batch_output = shapes[OUTPUT_INDEX][0];
    int channel_output = shapes[OUTPUT_INDEX][1];
    int height_output = shapes[OUTPUT_INDEX][2];
    int width_output = shapes[OUTPUT_INDEX][3];

    // fix at now ,need to be changed in future
    const int kernel_size = 1;
    const bool bilinear = true;

    int batch_stride_x1  = channel_x1 * height_x1 * width_x1;
    int channel_stride_x1  = height_x1 * width_x1;
    int height_stride_x1 = width_x1;
    int batch_stride_x2 = channel_x2 * height_x2 * width_x2;
    int channel_stride_x2  = height_x2 * width_x2;
    int height_stride_x2  = width_x2;
    size_t size = batch_output * channel_output * height_output * width_output;
    Resample2dInitKernel<<<size / CUDA_NUM_THREADS +1, CUDA_NUM_THREADS, 0, custream>>>(size, out_data);

    Resample2dKernel<<< (size + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, custream>>>
                   (size, x1, x2, out_data, batch_stride_x1, channel_stride_x1, height_stride_x1,
                   batch_stride_x2, channel_stride_x2, height_stride_x2, batch_output, channel_output,
                    height_output, width_output, kernel_size , bilinear);
    return 0;
}


__global__ void kernel_resample2d_grad_input1(size_t size,
    const float* input1, int batch_input1, int channel_input1, int height_input1, int width_input1,
    const float* input2, int batch_stride_input2, int channel_stride_input2, int height_stride_input2,
    const float* gradOutput, int batch_gradOutput, int channel_gradOutput, int height_gradOutput, int width_gradOutput,
    int batch_stride_gradOutput, int channel_stride_gradOutput, int height_stride_gradOutput,
    float* gradInput, int batch_stride_gradInput, int channel_stride_gradInput, int height_stride_gradInput,
    int kernel_size, bool bilinear) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    int dim_b = batch_gradOutput;
    int dim_c = channel_gradOutput;
    int dim_h = height_gradOutput;
    int dim_w = width_gradOutput;
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = (index / dim_chw) % dim_b;
    int c = (index / dim_hw)  % dim_c;
    int y = (index / dim_w)   % dim_h;
    int x = (index)  % dim_w;

    float dx = DIM3_INDEX(input2, b, 0, y, x, batch_stride_input2, channel_stride_input2, height_stride_input2);
    float dy = DIM3_INDEX(input2, b, 1, y, x, batch_stride_input2, channel_stride_input2, height_stride_input2);

    float xf = x + dx;
    float yf = y + dy;
    float alpha = xf - static_cast<int>(xf);  // alpha
    float beta = yf - static_cast<int>(yf);  // beta

    int idim_h = height_input1;
    int idim_w = width_input1;

    int xL = max(min(static_cast<int>(floor(xf)),    idim_w-1), 0);
    int xR = max(min(static_cast<int>(floor(xf)+1), idim_w -1), 0);
    int yT = max(min(static_cast<int>(floor(yf)),    idim_h-1), 0);
    int yB = max(min(static_cast<int>(floor(yf)+1),  idim_h-1), 0);

    float w1, w2, w3, w4;
    float num = 1.f;
    w1 = (num-alpha)*(num-beta);
    w2 = (alpha)*(num-beta);
    w3 = (num-alpha)*(beta);
    w4 = (alpha)*(beta);

    float gradnum = DIM3_INDEX(gradOutput, b, c, y, x,
            batch_stride_gradOutput, channel_stride_gradOutput, height_stride_gradOutput);
    for (int fy = 0; fy < kernel_size; fy += 1) {
        for (int fx = 0; fx < kernel_size; fx += 1) {
            int indexTL = GET_INDEX(b, c, (yT + fy), (xL + fx),
                        batch_stride_gradInput, channel_stride_gradInput, height_stride_gradInput);
            MsAtomicAdd(&gradInput[indexTL], w1 * gradnum);

            int indexTR = GET_INDEX(b, c, (yT + fy), (xR + fx),
                        batch_stride_gradInput, channel_stride_gradInput, height_stride_gradInput);
            MsAtomicAdd(&gradInput[indexTR], w2 * gradnum);

            int indexBL = GET_INDEX(b, c, (yB + fy), (xL + fx),
                        batch_stride_gradInput, channel_stride_gradInput, height_stride_gradInput);
            MsAtomicAdd(&gradInput[indexBL], w3 * gradnum);

            int indexBR = GET_INDEX(b, c, (yB + fy), (xR + fx),
                        batch_stride_gradInput, channel_stride_gradInput, height_stride_gradInput);
            MsAtomicAdd(&gradInput[indexBR], w4 * gradnum);
        }
    }
}


__global__ void kernel_resample2d_grad_input2(size_t size,
    const float  *input1, int batch_stride_input1, int channel_stride_input1, int height_stride_input1,
    const float  *input2, int batch_stride_input2, int channel_stride_input2, int height_stride_input2,
    const float  *gradOutput, int channel_gradOutput, int batch_stride_gradOutput,
    int channel_stride_gradOutput, int height_stride_gradOutput,
    float  *gradInput, int batch_gradInput, int channel_gradInput, int height_gradInput, int width_gradInput,
    int batch_stride_gradInput, int channel_stride_gradInput, int height_stride_gradInput,
    int kernel_size, bool bilinear) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    float output = 0.0;
    int kernel_rad = (kernel_size - 1)/2;

    int dim_b = batch_gradInput;
    int dim_c = channel_gradInput;
    int dim_h = height_gradInput;
    int dim_w = width_gradInput;
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = (index / dim_chw) % dim_b;
    int c = (index / dim_hw)  % dim_c;
    int y = (index / dim_w)   % dim_h;
    int x = (index)  % dim_w;

    int odim_c = channel_gradOutput;

    float dx = DIM3_INDEX(input2, b, 0, y, x, batch_stride_input2, channel_stride_input2, height_stride_input2);
    float dy = DIM3_INDEX(input2, b, 1, y, x, batch_stride_input2, channel_stride_input2, height_stride_input2);

    float xf = x + dx;
    float yf = y + dy;

    int xL = max(min(static_cast<int>(floor(xf)),    dim_w-1), 0);
    int xR = max(min(static_cast<int>(floor(xf)+1), dim_w -1), 0);
    int yT = max(min(static_cast<int>(floor(yf)),    dim_h-1), 0);
    int yB = max(min(static_cast<int>(floor(yf)+1),  dim_h-1), 0);

    if (c % 2) {
        float gamma = 1 - (xf - floor(xf));  // alpha
        for (int i = 0; i <= 2*kernel_rad ; ++i) {
            for (int j = 0; j <= 2*kernel_rad; ++j) {
                for (int ch = 0; ch < odim_c; ++ch) {
                    float gradout = DIM3_INDEX(gradOutput, b, ch, y, x,
                     batch_stride_gradOutput, channel_stride_gradOutput, height_stride_gradOutput);
                    output += (gamma) * gradout * DIM3_INDEX(input1, b, ch, (yB + j), (xL + i),
                                batch_stride_input1, channel_stride_input1, height_stride_input1);
                    output -= (gamma) * gradout * DIM3_INDEX(input1, b, ch, (yT + j), (xL + i),
                                batch_stride_input1, channel_stride_input1, height_stride_input1);
                    output += (1-gamma) * gradout * DIM3_INDEX(input1, b, ch, (yB + j), (xR + i),
                                batch_stride_input1, channel_stride_input1, height_stride_input1);
                    output -= (1-gamma) * gradout * DIM3_INDEX(input1, b, ch, (yT + j), (xR + i),
                                batch_stride_input1, channel_stride_input1, height_stride_input1);
                }
            }
        }
    } else {
        float gamma = 1 - (yf - floor(yf));  // alpha
        for (int i = 0; i <= 2*kernel_rad; ++i) {
            for (int j = 0; j <= 2*kernel_rad; ++j) {
                for (int ch = 0; ch < odim_c; ++ch) {
                    float gradout = static_cast<float>(DIM3_INDEX(gradOutput, b, ch, y, x,
                                batch_stride_gradOutput, channel_stride_gradOutput, height_stride_gradOutput));
                    output += (gamma) * gradout * static_cast<float>(DIM3_INDEX(input1, b, ch, (yT + j), (xR + i),
                                batch_stride_input1, channel_stride_input1, height_stride_input1));
                    output -= (gamma)* gradout * static_cast<float>(DIM3_INDEX(input1, b, ch, (yT + j), (xL + i),
                                batch_stride_input1, channel_stride_input1, height_stride_input1));
                    output += (1-gamma)* gradout * static_cast<float>(DIM3_INDEX(input1, b, ch, (yB + j), (xR + i),
                                batch_stride_input1, channel_stride_input1, height_stride_input1));
                    output -= (1-gamma) * gradout * static_cast<float>(DIM3_INDEX(input1, b, ch, (yB + j), (xL + i),
                                batch_stride_input1, channel_stride_input1, height_stride_input1));
                }
            }
        }
    }
    gradInput[index] = output;
}


extern "C" int Resample2dGrad(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                              void *stream, void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    constexpr int INPUT1_INDEX = 0;
    constexpr int INPUT2_INDEX = 1;
    constexpr int GRAD_OUTPUT_INDEX = 2;
    constexpr int TOTAL_PARAM_NUM = 5;

    if (nparam != TOTAL_PARAM_NUM) {
        return 1;
    }
    // This is to check if the type of parameters the same as what the user wants.
    for (int i = 0; i < nparam; i++) {
        if (strcmp(dtypes[i], "float32") != 0) {
            return 2;
        }
    }

    float *x1 = static_cast<float *>(params[0]);
    float *x2 = static_cast<float *>(params[1]);
    float *dout = static_cast<float *>(params[2]);
    float *dx1 = static_cast<float *>(params[3]);
    float *dx2 = static_cast<float *>(params[4]);

    int batch_x1 = shapes[INPUT1_INDEX][0];
    int channel_x1 = shapes[INPUT1_INDEX][1];
    int height_x1 = shapes[INPUT1_INDEX][2];
    int width_x1 = shapes[INPUT1_INDEX][3];

    int batch_x2 = shapes[INPUT2_INDEX][0];
    int channel_x2 = shapes[INPUT2_INDEX][1];
    int height_x2 = shapes[INPUT2_INDEX][2];
    int width_x2 = shapes[INPUT2_INDEX][3];

    int batch_dout = shapes[GRAD_OUTPUT_INDEX][0];
    int channel_dout = shapes[GRAD_OUTPUT_INDEX][1];
    int height_dout = shapes[GRAD_OUTPUT_INDEX][2];
    int width_dout = shapes[GRAD_OUTPUT_INDEX][3];

    // fix at now ,need to be changed in future
    const int kernel_size = 1;
    const bool bilinear = true;

    int batch_dx1  = batch_x1;
    int channel_dx1  = channel_x1;
    int height_dx1  = height_x1;
    int width_dx1 = width_x1;
    int batch_dx2  = batch_x2;
    int channel_dx2 = channel_x2;
    int height_dx2 = height_x2;
    int width_dx2 = width_x2;
    int batch_stride_x1  = channel_x1 * height_x1 * width_x1;
    int channel_stride_x1  = height_x1 * width_x1;
    int height_stride_x1  = width_x1;
//     int width_stride_x1 = 1;
    int batch_stride_x2  = channel_x2 * height_x2 * width_x2;
    int channel_stride_x2  = height_x2 * width_x2;
    int height_stride_x2 = width_x2;
//     int width_stride_x2 = 1;
    int batch_stride_dx1  = batch_stride_x1;
    int channel_stride_dx1 = channel_stride_x1;
    int height_stride_dx1 = height_stride_x1;
//     int width_stride_dx1 = width_stride_x1;
    int batch_stride_dx2  = batch_stride_x2;
    int channel_stride_dx2 = channel_stride_x2;
    int height_stride_dx2 = height_stride_x2;
//     int width_stride_dx2 = width_stride_x2;
    int batch_stride_dout  = channel_dout * height_dout * width_dout;
    int channel_stride_dout  = height_dout * width_dout;
    int height_stride_dout  = width_dout;
//     int width_stride_dout  = 1;

    size_t dx1_size = batch_dx1 * channel_dx1 * height_dx1 * width_dx1;

    Resample2dInitKernel<<<dx1_size / CUDA_NUM_THREADS +1, CUDA_NUM_THREADS, 0, custream>>>(dx1_size, dx1);
    size_t dx2_size = batch_dx2 * channel_dx2 * height_dx2 * width_dx2;
    Resample2dInitKernel<<<dx2_size / CUDA_NUM_THREADS +1, CUDA_NUM_THREADS, 0, custream>>>(dx2_size, dx2);

    size_t dout_size = batch_dout * channel_dout * height_dout * width_dout;

    kernel_resample2d_grad_input1<<<(dout_size + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS,
                                     0, custream>>>(dout_size,
                                     x1, batch_x1, channel_x1, height_x1, width_x1,
                                     x2, batch_stride_x2, channel_stride_x2, height_stride_x2,
                                     dout, batch_dout, channel_dout, height_dout, width_dout,
                                     batch_stride_dout, channel_stride_dout, height_stride_dout,
                                     dx1, batch_stride_dx1, channel_stride_dx1, height_stride_dx1,
                                     kernel_size, bilinear);

    kernel_resample2d_grad_input2<<<(dx2_size + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS,
                                    0, custream>>>(dx2_size,
                                     x1, batch_stride_x1, channel_stride_x1, height_stride_x1,
                                     x2, batch_stride_x2, channel_stride_x2, height_stride_x2,
                                     dout, channel_dout, batch_stride_dout, channel_stride_dout, height_stride_dout,
                                     dx2, batch_dx2, channel_dx2, height_dx2, width_dx2,
                                     batch_stride_dx2, channel_stride_dx2, height_stride_dx2,
                                     kernel_size, bilinear);
    return 0;
}

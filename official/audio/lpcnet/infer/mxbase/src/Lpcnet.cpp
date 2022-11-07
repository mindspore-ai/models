/*
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Lpcnet.h"
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>
#include <numeric>
#include <typeinfo>
#include "acl/acl.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

namespace {
    const int FRAME_SIZE = 160;
    const int NB_FEATURES = 36;
    const int NB_USED_FEATURES = 20;
    const int ORDER = 16;
    const int RNN_UNITS1 = 384;
    const int RNN_UNITS2 = 16;
    const float scale = 255.0/32768.0;
    const float scale_1 = 32768.0/255.0;
    const int feature_chunk_size = 500;
    const int pcm_chunk_size = FRAME_SIZE * feature_chunk_size;

    double ulaw2lin(int u) {
        u = u - 128;
        int s;
        if (u < 0) {
            s = -1;
        } else if (u > 0) {
            s = 1;
        } else {
            s = 0;
        }
        u = abs(u);
        return s * scale_1 * (expm1(u / 128. * log(256)));
    }

    int16_t lin2ulaw(double x) {
        int s;
        if (x < 0) {
            s = -1;
        } else if (x > 0) {
            s = 1;
        } else {
            s = 0;
        }
        x = std::abs(x);
        double u = (s * (128 * log1p(scale * x) / log(256)));
        if (128 + round(u) < 0) {
            u = 0;
        } else if (128 + round(u) > 255) {
            u = 255;
        } else {
            u = 128 + round(u);
        }
        return static_cast<int16_t>(u);
    }
    int multinomial(double *x, uint32_t seed) {
        double val = static_cast<double>(rand_r(&seed)) / RAND_MAX;
        double sum_x = 0;
        for (int i = 0; i < 256; i++) {
            sum_x += x[i];
            if (val <= sum_x) {
                return i;
            }
        }
    }
}  // namespace

APP_ERROR Lpcnet::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    model_encoder = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_encoder->Init(initParam.encoder_modelPath, modelDesc_encoder);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    model_decoder = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_decoder->Init(initParam.decoder_modelPath, modelDesc_decoder);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Lpcnet::DeInit() {
    model_encoder->DeInit();
    model_decoder->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}
void Lpcnet::VectorToTensorBase_Int32(const std::vector<std::vector<int>> &batchFeatureVector,
                                           MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 1;
    std::vector<uint32_t> shape = {};
    shape.push_back(1);
    shape.push_back(batchFeatureVector.size());
    shape.push_back(batchFeatureVector[0].size());
    for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
    }

    int *metaFeatureData = new int[dataSize];
    uint32_t idx = 0;
    for (size_t bs = 0; bs < batchFeatureVector.size(); bs++) {
        for (size_t d = 0; d < batchFeatureVector[bs].size(); d++) {
            metaFeatureData[idx++] = batchFeatureVector[bs][d];
        }
    }
    MxBase::MemoryData memoryDataDst(dataSize * 4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(metaFeatureData, dataSize * 4, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return;
    }

    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_INT32);
}


void Lpcnet::VectorToTensorBase_Float32(const std::vector<std::vector<float>> &batchFeatureVector,
                                        MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 1;
    std::vector<uint32_t> shape = {};
    shape.push_back(1);
    shape.push_back(batchFeatureVector.size());
    shape.push_back(batchFeatureVector[0].size());
    for (uint32_t s = 0; s < shape.size(); ++s) {
            dataSize *= shape[s];
        }
    float *metaFeatureData = new float[dataSize];
    uint32_t idx = 0;
    for (size_t bs = 0; bs < batchFeatureVector.size(); bs++) {
        for (size_t d = 0; d < batchFeatureVector[bs].size(); d++) {
            metaFeatureData[idx++] = batchFeatureVector[bs][d];
        }
    }
    MxBase::MemoryData memoryDataDst(dataSize*4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(metaFeatureData, dataSize*4, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return;
    }
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
}

APP_ERROR Lpcnet::Inference_Encoder(const std::vector<MxBase::TensorBase> &inputs,
                                    std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_encoder->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_encoder.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_encoder.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_encoder.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_encoder->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference encoder failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Lpcnet::Inference_Decoder(std::vector<MxBase::TensorBase> &inputs,
                                    std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_decoder->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_decoder.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_decoder.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_decoder.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_decoder->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference decoder failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

std::vector<std::vector<float>> Lpcnet::make_zero_martix_float(int m, int n) {
    std::vector<std::vector<float>> array;
    std::vector<float> temparay;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            temparay.push_back(0);
        array.push_back(temparay);
        temparay.erase(temparay.begin(), temparay.end());
    }
    return array;
}

std::vector<std::vector<double>> Lpcnet::make_zero_martix_double(int m, int n) {
    std::vector<std::vector<double>> array;
    std::vector<double> temparay;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            temparay.push_back(0);
        array.push_back(temparay);
        temparay.erase(temparay.begin(), temparay.end());
    }
    return array;
}

std::vector<std::vector<int>> Lpcnet::make_128_martix_int(int m, int n) {
    std::vector<std::vector<int>> array;
    std::vector<int> temparay;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            temparay.push_back(128);
        array.push_back(temparay);
        temparay.erase(temparay.begin(), temparay.end());
    }
    return array;
}

void Lpcnet::process_decoder(double *new_p, const std::vector<std::vector<float>> &feature, int fr) {
    for (int j = 0; j < 256; j++) {
        new_p[j] *= pow(new_p[j], std::max(0.0, 1.5 * feature[fr][19] - 0.5));
    }
    double sum_p = 0;
    for (int j = 0; j < 256; j++) {
        sum_p += new_p[j];
    }
    for (int j = 0; j < 256; j++) {
        new_p[j] = new_p[j] / (1e-18 + sum_p) - 0.002;
        if (new_p[j] < 0) {
            new_p[j] = 0.0;
        }
    }
    sum_p = 0;
    for (int j = 0; j < 256; j++) {
        sum_p += new_p[j];
    }
    for (int j = 0; j < 256; j++) {
        new_p[j] = new_p[j] / (1e-8 + sum_p);
    }
}

APP_ERROR Lpcnet::Process(const std::vector<std::vector<float>> &cfeat, const std::vector<std::vector<int>> &period,
                          const std::vector<std::vector<float>> &feature, const InitParam &initParam,
                          std::vector<int16_t> &mem_outputs) {
    std::vector<MxBase::TensorBase> encoder_inputs = {};
    std::vector<MxBase::TensorBase> encoder_outputs = {};
    float mem = 0.0;
    float coef = 0.85;
    MxBase::TensorBase tensorBase1, tensorBase2, tensorBaseft;
    VectorToTensorBase_Float32(cfeat, tensorBase1);
    encoder_inputs.push_back(tensorBase1);
    VectorToTensorBase_Int32(period, tensorBase2);
    encoder_inputs.push_back(tensorBase2);
    VectorToTensorBase_Float32(feature, tensorBaseft);
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = Inference_Encoder(encoder_inputs, encoder_outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs_encoder = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs_encoder;
    if (!encoder_outputs[0].IsHost()) {
        encoder_outputs[0].ToHost();
    }
    float *enc_result = reinterpret_cast<float *>(encoder_outputs[0].GetBuffer());
    std::vector<std::vector<double>> pcm = make_zero_martix_double(1, pcm_chunk_size);
    std::vector<std::vector<int>> fexc = make_128_martix_int(1, 3);
    std::vector<std::vector<float>> state1 = make_zero_martix_float(1, RNN_UNITS1);
    std::vector<std::vector<float>> state2 = make_zero_martix_float(1, RNN_UNITS2);
    int skip = ORDER + 1;
    startTime = std::chrono::high_resolution_clock::now();
    for (int fr = 0; fr < feature_chunk_size; fr++) {
        std::vector<float> a, enc_cfeat;
        std::vector<std::vector<float>> enc_Cfeat;
        for (int i = 20; i < 36; i++) {
            a.push_back(feature[fr][i]);
        }
        for (int i = fr * 128; i < (fr + 1) * 128; i++) {
            enc_cfeat.push_back(enc_result[i]);
        }
        enc_Cfeat.push_back(enc_cfeat);
        for (int i = skip; i < FRAME_SIZE; i++) {
            unsigned int seed = static_cast<uint32_t>(i);
            std::vector<MxBase::TensorBase> decoder_inputs = {};
            std::vector<MxBase::TensorBase> decoder_outputs = {};
            std::vector<double> tmp;
            int k = 0;
            for (int j = fr * FRAME_SIZE + i - 1; j > fr * FRAME_SIZE + i - ORDER - 1; j--, k++) {
                tmp.push_back(pcm[0][j] * static_cast<double>(a[k]));
            }
            double pred = -(accumulate(tmp.begin(), tmp.end(), 0));
            fexc[0][1] = static_cast<int>(lin2ulaw(pred));
            MxBase::TensorBase decoder_input1, decoder_input2, decoder_input3, decoder_input4;
            VectorToTensorBase_Int32(fexc, decoder_input1);
            decoder_inputs.push_back(decoder_input1);
            VectorToTensorBase_Float32(enc_Cfeat, decoder_input2);
            decoder_inputs.push_back(decoder_input2);
            VectorToTensorBase_Float32(state1, decoder_input3);
            decoder_inputs.push_back(decoder_input3);
            VectorToTensorBase_Float32(state2, decoder_input4);
            decoder_inputs.push_back(decoder_input4);
            ret = Inference_Decoder(decoder_inputs, decoder_outputs);
            for (auto retTensor : decoder_outputs) {
                if (!retTensor.IsHost()) {
                    retTensor.ToHost();
                }
            }
            float *p = reinterpret_cast<float *>(decoder_outputs[0].GetBuffer());
            float *state1_ = reinterpret_cast<float *>(decoder_outputs[1].GetBuffer());
            float *state2_ = reinterpret_cast<float *>(decoder_outputs[2].GetBuffer());
            for (int j = 0; j < RNN_UNITS1; j++) {
                state1[0][j] = state1_[j];
            }
            for (int j = 0; j < RNN_UNITS2; j++) {
                state2[0][j] = state2_[j];
            }
            double new_p[256];
            for (int j = 0; j < 256; j++) {
                new_p[j] = static_cast<double>(p[j]);
            }
            process_decoder(new_p, feature, fr);
            fexc[0][2] = multinomial(new_p, seed);
            pcm[0][fr * FRAME_SIZE + i] = pred + ulaw2lin(fexc[0][2]);
            fexc[0][0] = static_cast<int>(lin2ulaw(pcm[0][fr * FRAME_SIZE + i]));
            mem = coef * mem + pcm[0][fr * FRAME_SIZE + i];
            mem_outputs.push_back(static_cast<int16_t>(mem));
        }
        skip = 0;
    }
    endTime = std::chrono::high_resolution_clock::now();
    double costMs_decoder = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs_decoder;
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
}

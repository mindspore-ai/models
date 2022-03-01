/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "inc/model_process.h"
#include <iostream>
#include <map>
#include <sstream>
#include <algorithm>
#include <functional>
#include <chrono>
#include <ctime>
#include <exception>
#include <stdexcept>

#include "inc/utils.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

extern bool g_isDevice;

ModelProcess::ModelProcess() :modelId_(0), modelWorkSize_(0), modelWeightSize_(0), modelWorkPtr_(nullptr),
modelWeightPtr_(nullptr), loadFlag_(false), modelDesc_(nullptr), input_(nullptr), output_(nullptr) {
}

ModelProcess::~ModelProcess() {
    UnloadModel();
    DestroyModelDesc();
    DestroyInput();
    DestroyOutput();
}

Result ModelProcess::LoadModel(const char *modelPath) {
    if (loadFlag_) {
        ERROR_LOG("model has already been loaded");
        return FAILED;
    }

    aclError ret = aclmdlQuerySize(modelPath, &modelWorkSize_, &modelWeightSize_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("query model failed, model file is %s, errorCode is %d",
        modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }
    // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred to use
    // and huge memory can improve performance.
    ret = aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("malloc buffer for work failed, require size is %zu, errorCode is %d",
        modelWorkSize_, static_cast<int32_t>(ret));
        return FAILED;
    }

    // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred to use
    // and huge memory can improve performance.
    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("malloc buffer for weight failed, require size is %zu, errorCode is %d",
        modelWeightSize_, static_cast<int32_t>(ret));
        return FAILED;
    }

    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelWorkPtr_,
    modelWorkSize_, modelWeightPtr_, modelWeightSize_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("load model from file failed, model file is %s, errorCode is %d",
        modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }

    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath);
    return SUCCESS;
}

Result ModelProcess::CreateModelDesc() {
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("get model description failed, modelId is %u, errorCode is %d",
        modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

void ModelProcess::DestroyModelDesc() {
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
    INFO_LOG("destroy model description success");
}

size_t ModelProcess::GetInputSizeByIndex(const size_t index) {
    if (modelDesc_ == nullptr) {
        throw std::runtime_error("ModelProcess::getInputNameByIndex exception: modelDesc_ is nullptr");
    }
    auto num_inputs = getNumInputs();
    if (index >= num_inputs) {
        std::stringstream ss;
        ss << "ModelProcess::getInputNameByIndex exception: input index "
           << index << " is less than the number of model inputs " << num_inputs;
        string errorstr = ss.str();
        throw std::invalid_argument(errorstr);
    }
    return aclmdlGetInputSizeByIndex(modelDesc_, index);
}

Result ModelProcess::CreateInput() {
    // om used in this sample has only one input
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }

    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::AddInput(void *data, size_t dataSize, int index) {
    // om used in this sample has only one input
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }

    // om used in this sample has only one input
    if (input_ == nullptr) {
        ERROR_LOG("input_ is nullptr, create input first");
        return FAILED;
    }

    size_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, index);
    if (dataSize != modelInputSize) {
        ERROR_LOG("Input %d: input image size[%zu] is not equal to model input size[%zu]",
        index, dataSize, modelInputSize);
        return FAILED;
    }

    aclDataBuffer *inputData = aclCreateDataBuffer(data, dataSize);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }

    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
        (void)aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return FAILED;
    }
    INFO_LOG("create model input %d success", index);

    return SUCCESS;
}

void ModelProcess::DestroyInput() {
    if (input_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        (void)aclDestroyDataBuffer(dataBuffer);
    }
    (void)aclmdlDestroyDataset(input_);
    input_ = nullptr;
    INFO_LOG("destroy model input success");
}

Result ModelProcess::CreateOutput() {
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create output failed");
        return FAILED;
    }

    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }

    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    INFO_LOG("CreateOutput: model has %zd outputs", outputSize);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);

        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, modelOutputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed, errorCode is %d",
            modelOutputSize, static_cast<int32_t>(ret));
            return FAILED;
        }

        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, modelOutputSize);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            (void)aclrtFree(outputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("can't add data buffer, create output failed, errorCode is %d",
            static_cast<int32_t>(ret));
            (void)aclrtFree(outputBuffer);
            (void)aclDestroyDataBuffer(outputData);

            return FAILED;
        }
    }

    INFO_LOG("create model output success");

    return SUCCESS;
}

void ModelProcess::DumpModelOutputResult(const std::string& path) {
    size_t outputNum = aclmdlGetNumOutputs(modelDesc_);
    INFO_LOG("Model has [%zd] outputs", outputNum);
    for (size_t i = 0; i < outputNum; ++i) {
        std::stringstream ss;
        ss << path << "/" << "output" << "_" << i << ".bin";
        std::string outputFileName = ss.str();
        FILE *outputFile = fopen(outputFileName.c_str(), "wb");
        if (outputFile != nullptr) {
            // get model output data
            aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output_, i);
            void *data = aclGetDataBufferAddr(dataBuffer);
            uint32_t len = aclGetDataBufferSizeV2(dataBuffer);

            void *outHostData = nullptr;
            aclError ret = ACL_ERROR_NONE;
            if (!g_isDevice) {
                ret = aclrtMallocHost(&outHostData, len);
                if (ret != ACL_ERROR_NONE) {
                    ERROR_LOG("aclrtMallocHost failed, malloc len[%u], errorCode[%d]",
                    len, static_cast<int32_t>(ret));
                    fclose(outputFile);
                    return;
                }

                // if app is running in host, need copy model output data from device to host
                ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_ERROR_NONE) {
                    ERROR_LOG("aclrtMemcpy failed, errorCode[%d]", static_cast<int32_t>(ret));
                    (void)aclrtFreeHost(outHostData);
                    fclose(outputFile);
                    return;
                }

                fwrite(outHostData, len, sizeof(char), outputFile);

                ret = aclrtFreeHost(outHostData);
                if (ret != ACL_ERROR_NONE) {
                    ERROR_LOG("aclrtFreeHost failed, errorCode[%d]", static_cast<int32_t>(ret));
                    fclose(outputFile);
                    return;
                }
            } else {
                // if app is running in host, write model output data into result file
                fwrite(data, len, sizeof(char), outputFile);
            }
            fclose(outputFile);
        } else {
            ERROR_LOG("create output file [%s] failed", outputFileName.c_str());
            return;
        }
        INFO_LOG("Result is saved to [%s]", outputFileName.c_str());
    }

    INFO_LOG("dump data success");
    return;
}

void ModelProcess::OutputModelResult() {
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        // get model output data
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);

        void *outHostData = nullptr;
        aclError ret = ACL_ERROR_NONE;
        float *outData = nullptr;
        if (!g_isDevice) {
            ret = aclrtMallocHost(&outHostData, len);
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("aclrtMallocHost failed, malloc len[%u], errorCode[%d]",
                len, static_cast<int32_t>(ret));
                return;
            }

            // if app is running in host, need copy model output data from device to host
            ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("aclrtMemcpy failed, errorCode[%d]", static_cast<int32_t>(ret));
                (void)aclrtFreeHost(outHostData);
                return;
            }

            outData = reinterpret_cast<float*>(outHostData);
        } else {
            outData = reinterpret_cast<float*>(data);
        }
        std::map<float, unsigned int, std::greater<float> > resultMap;
        for (unsigned int j = 0; j < len / sizeof(float); ++j) {
            resultMap[*outData] = j;
            outData++;
        }

        int cnt = 0;
        for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
            // print top 5
            if (++cnt > 5) {
                break;
            }

            INFO_LOG("top %d: index[%d] value[%lf]", cnt, it->second, it->first);
        }
        if (!g_isDevice) {
            ret = aclrtFreeHost(outHostData);
            if (ret != ACL_ERROR_NONE) {
                ERROR_LOG("aclrtFreeHost failed, errorCode[%d]", static_cast<int32_t>(ret));
                return;
            }
        }
    }

    INFO_LOG("output data success");
    return;
}

void ModelProcess::DestroyOutput() {
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
    INFO_LOG("destroy model output success");
}

Result ModelProcess::Execute() {
    auto begin_time = high_resolution_clock::now();
    aclError ret = aclmdlExecute(modelId_, input_, output_);
    auto end_time = high_resolution_clock::now();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("execute model failed, modelId is %u, errorCode is %d",
        modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    duration<double, std::milli> run_time = end_time - begin_time;
    INFO_LOG("model execute success");
    INFO_LOG("Execute time %f ms.", static_cast<float>(run_time.count()));
    return SUCCESS;
}

void ModelProcess::UnloadModel() {
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("unload model failed, modelId is %u, errorCode is %d",
        modelId_, static_cast<int32_t>(ret));
    }

    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelWorkPtr_ != nullptr) {
        (void)aclrtFree(modelWorkPtr_);
        modelWorkPtr_ = nullptr;
        modelWorkSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        (void)aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
    modelId_ = 0;
}

size_t ModelProcess::getNumInputs() const {
    if (modelDesc_ != nullptr) {
        return aclmdlGetNumInputs(modelDesc_);
    }
    return -1;
}

std::string ModelProcess::getInputNameByIndex(size_t index) const {
    if (modelDesc_ == nullptr) {
        throw std::runtime_error("ModelProcess::getInputNameByIndex exception: modelDesc_ is nullptr");
    }
    auto num_inputs = getNumInputs();
    if (index >= num_inputs) {
        std::stringstream ss;
        ss << "ModelProcess::getInputNameByIndex exception: input index "
           << index << " is less than the number of model inputs " << num_inputs;
        std::string errorstr = ss.str();
        throw std::invalid_argument(errorstr);
    }
    return std::string(aclmdlGetInputNameByIndex(modelDesc_, index));
}

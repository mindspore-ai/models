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

#include "inc/sample_process.h"
#include <iostream>
#include <chrono>
#include <array>
#include <sstream>
#include "inc/model_process.h"
#include "acl/acl.h"
#include "inc/utils.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

extern bool g_isDevice;

SampleProcess::SampleProcess() :deviceId_(0), context_(nullptr), stream_(nullptr),
resources_base_path_(""), om_path_(""), output_path_(""), acl_path_("") {
}

SampleProcess::~SampleProcess() {
    DestroyResource();
}

void SampleProcess::setResourcesBasePath(const std::string& path) {
    resources_base_path_ = path;
}

void SampleProcess::setOmPath(const std::string& path) {
    om_path_ = path;
}

void SampleProcess::setOutputPath(const std::string& path) {
    output_path_ = path;
}

void SampleProcess::setAclPath(const std::string& path) {
    acl_path_ = path;
}

Result SampleProcess::InitResource() {
    // ACL init
    const char *aclConfigPath = acl_path_.c_str();
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("acl init success");

    // set device
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl set device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("set device %d success", deviceId_);

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create context failed, deviceId = %d, errorCode = %d",
        deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed, deviceId = %d, errorCode = %d",
        deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("create stream success");

    // get run mode
    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("get run mode success");
    return SUCCESS;
}

void SampleProcess::freeAllBuffers(std::vector<void*>* all_buffers_ptr) {
    std::vector<void*>& allBuffers = *all_buffers_ptr;
    for (size_t i = 0; i < allBuffers.size(); ++i) {
        aclrtFree(allBuffers[i]);
    }
}

Result SampleProcess::RunInferenceLoop(ModelProcess* modelProcessPtr,
                                       std::vector<void*>* all_buffers_ptr,
                                       const std::vector<size_t>& buffer_sizes,
                                       const std::vector<std::string>& all_dirs,
                                       size_t num_inputs) {
    uint32_t run_counter = 0;
    double total_run_time = 0.0;
    double total_e2e_time = 0.0;
    ModelProcess& modelProcess = *modelProcessPtr;
    std::vector<void*>& all_buffers = *all_buffers_ptr;

    for (size_t index = 0; index < all_dirs.size(); ++index) {
        auto start_e2e_time = high_resolution_clock::now();
        Result ret;
        for (size_t input_index = 0; input_index < num_inputs; ++input_index) {
            std::stringstream ss;
            ss << resources_base_path_ + "/" << all_dirs[index] << "/input_" << input_index << ".bin";
            auto filename = ss.str();
            INFO_LOG("start to process file:%s", filename.c_str());
            // copy image data to device buffer
            ret = Utils::MemcpyFileToDeviceBuffer(filename, all_buffers[input_index], buffer_sizes[input_index]);

            if (ret != SUCCESS) {
                freeAllBuffers(all_buffers_ptr);
                ERROR_LOG("memcpy device buffer failed, index is %zu", input_index);
                return FAILED;
            }
        }

        auto start_run_time = high_resolution_clock::now();
        ret = modelProcess.Execute();
        auto end_run_time = high_resolution_clock::now();
        if (ret != SUCCESS) {
            ERROR_LOG("execute inference failed");
            freeAllBuffers(all_buffers_ptr);
            return FAILED;
        }

        // Print the top 5 confidence values with indexes.
        // Use function [DumpModelOutputResult] if you want to dump results to file in the current directory.
        //        modelProcess.OutputModelResult();
        auto out_path = output_path_;
        if (out_path[out_path.size() - 1] != '/') {
            out_path += '/';
        }
        out_path += all_dirs[index];
        Utils::mkdir(out_path);
        modelProcess.DumpModelOutputResult(out_path);
        auto end_e2e_time = high_resolution_clock::now();
        run_counter++;
        duration<double, std::milli> run_time = end_run_time - start_run_time;
        duration<double, std::milli> e2e_time = end_e2e_time - start_e2e_time;
        total_run_time += run_time.count();
        total_e2e_time += e2e_time.count();
    }

    INFO_LOG("Average run time over %d runs: %f ms", run_counter, total_run_time/run_counter);
    INFO_LOG("Average e2e time over %d runs: %f ms", run_counter, total_e2e_time/run_counter);
    return SUCCESS;
}

Result SampleProcess::Process() {
    // model init
    ModelProcess modelProcess;
    const string modelFile = om_path_;
    const char* omModelPath = modelFile.c_str();
    Result ret = modelProcess.LoadModel(omModelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModel failed");
        return FAILED;
    }

    ret = modelProcess.CreateModelDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateModelDesc failed");
        return FAILED;
    }

    auto num_inputs = modelProcess.getNumInputs();
    INFO_LOG("Model has %zu inputs", num_inputs);

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = modelProcess.getInputNameByIndex(i);
        auto input_size = modelProcess.GetInputSizeByIndex(i);
        std::cout << "Model input " << i << ": " << input_name << "; size: " << input_size << " bytes" << std::endl;
    }

    auto all_dirs = Utils::ListDir(resources_base_path_);

    std::vector<void*> all_buffers;
    std::vector<size_t> buffer_sizes;

    ret = modelProcess.CreateInput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        freeAllBuffers(&all_buffers);
        return FAILED;
    }

    for (size_t i = 0; i < num_inputs; ++i) {
        void * inpBuffer = nullptr;
        size_t input_size = modelProcess.GetInputSizeByIndex(i);

        aclError aclRet = aclrtMalloc(&inpBuffer, input_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (aclRet != ACL_ERROR_NONE) {
            ERROR_LOG("malloc device buffer failed0. size is %zu, errorCode is %d",
            input_size, static_cast < int32_t > (aclRet));
            return FAILED;
        }

        all_buffers.push_back(inpBuffer);
        buffer_sizes.push_back(input_size);

        ret = modelProcess.AddInput(inpBuffer, input_size, i);
        if (ret != SUCCESS) {
            ERROR_LOG("execute AddInput failed");
            freeAllBuffers(&all_buffers);
            return FAILED;
        }
    }

    ret = modelProcess.CreateOutput();
    if (ret != SUCCESS) {
        freeAllBuffers(&all_buffers);
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    RunInferenceLoop(&modelProcess, &all_buffers, buffer_sizes, all_dirs, num_inputs);

    // release model input output
    modelProcess.DestroyInput();
    modelProcess.DestroyOutput();

    freeAllBuffers(&all_buffers);
    return SUCCESS;
}

void SampleProcess::DestroyResource() {
    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
    }
    INFO_LOG("end to reset device %d", deviceId_);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed, errorCode = %d", static_cast<int32_t>(ret));
    }
    INFO_LOG("end to finalize acl");
}

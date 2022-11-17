/*
 * Copyright(C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gflags/gflags.h>

#include <cstring>
#include <iostream>
#include <string>

#include <boost/property_tree/json_parser.hpp>

#include "MxBase/Log/Log.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxTools/Proto/MxpiDataTypeDeleter.h"


namespace {
const char STREAM_NAME[] = "im_deepfm";
}  // namespace

// the shape of feat_ids & feat_vals. It's defined by the inference model
const int COL_PER_SAMPLE = 39;

DEFINE_string(feat_ids, "../../data/feat_ids.bin", "feat_ids.");
DEFINE_string(feat_vals, "../../data/feat_vals.bin", "feat_vals.");
DEFINE_int32(sample_num, 2, "num of samples");
DEFINE_string(pipeline, "../../data/config/deepfm_ms.pipeline",
              "config file for this model.");

APP_ERROR ReadFileToMem(const std::string &filePath, std::string &mem) {
    std::ifstream file(filePath.c_str(), std::ifstream::binary);
    if (!file) {
        LogError << "open file for read error:" << filePath;
        return -1;
    }
    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);

    mem.resize(fileSize);
    file.read(&mem[0], fileSize);
    file.close();
    return APP_ERR_OK;
}

template <class Type>
int LoadBinaryFile(char *buffer, const std::string &filename, int struct_num,
                   size_t offset = 0) {
    std::ifstream rf(filename, std::ios::out | std::ios::binary);

    if (!rf) {
        LogError << "Cannot open file!";
        return -1;
    }

    if (offset > 0) {
        rf.seekg(sizeof(Type) * offset, rf.beg);
    }

    rf.read(buffer, sizeof(Type) * struct_num);
    rf.close();

    if (!rf.good()) {
        LogInfo << "Error occurred at reading time!";
        return -2;
    }

    return APP_ERR_OK;
}

template <class Type>
bool loadBinaryAsInput(
    std::string &mem, const std::string &filename,
    size_t struct_num = 1,     // number of <Type>s to read
    size_t offset = 0,         // pos to start reading, its unit is <Type>
    size_t input_index = 0) {  // index for the model inputs

    mem.resize(sizeof(Type) * struct_num);

    if (LoadBinaryFile<Type>(&mem[0], filename, struct_num, offset) !=
        APP_ERR_OK) {
        LogError << "load text:" << filename << " to device input[0] error!!!";
        return false;
    }

    return true;
}

// This function is only for reading sample txt.
APP_ERROR SendEachProtoBuf(
    int inPluginId, const void *dataPtr, int dataSize, int dataShape,
    int32_t dataType,
    std::shared_ptr<MxStream::MxStreamManager> &mxStreamManager,
    const std::string &streamName) {
    MxBase::MemoryData memorySrc(const_cast<void *>(dataPtr), dataSize,
                                 MxBase::MemoryData::MEMORY_HOST_NEW);
    MxBase::MemoryData memoryDst(dataSize, MxBase::MemoryData::MEMORY_HOST_NEW);

    APP_ERROR ret =
        MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, memorySrc);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to malloc and copy host memory.";
        return ret;
    }

    auto tensorPackageList = std::shared_ptr<MxTools::MxpiTensorPackageList>(
        new MxTools::MxpiTensorPackageList,
        MxTools::g_deleteFuncMxpiTensorPackageList);
    auto tensorPackage = tensorPackageList->add_tensorpackagevec();
    auto tensorVec = tensorPackage->add_tensorvec();
    tensorVec->set_tensordataptr((uint64_t)memoryDst.ptrData);
    tensorVec->set_tensordatasize(dataSize);
    tensorVec->set_tensordatatype(dataType);
    tensorVec->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);
    tensorVec->set_deviceid(0);
    tensorVec->add_tensorshape(1);
    tensorVec->add_tensorshape(dataShape);

    MxStream::MxstProtobufIn dataBuffer;
    std::ostringstream dataSource;
    dataSource << "appsrc" << inPluginId;
    dataBuffer.key = dataSource.str();
    dataBuffer.messagePtr =
        std::static_pointer_cast<google::protobuf::Message>(tensorPackageList);
    std::vector<MxStream::MxstProtobufIn> dataBufferVec;
    dataBufferVec.push_back(dataBuffer);

    ret = mxStreamManager->SendProtobuf(streamName, inPluginId, dataBufferVec);
    return ret;
}

void GetTensors(
    const std::shared_ptr<MxTools::MxpiTensorPackageList> &tensorPackageList,
    std::vector<MxBase::TensorBase> &tensors) {
    for (int i = 0; i < tensorPackageList->tensorpackagevec_size(); ++i) {
        for (int j = 0;
             j < tensorPackageList->tensorpackagevec(i).tensorvec_size(); j++) {
            MxBase::MemoryData memoryData = {};
            memoryData.deviceId =
                tensorPackageList->tensorpackagevec(i).tensorvec(j).deviceid();
            memoryData.type = (MxBase::MemoryData::MemoryType)tensorPackageList
                                  ->tensorpackagevec(i)
                                  .tensorvec(j)
                                  .memtype();
            memoryData.size = (uint32_t)tensorPackageList->tensorpackagevec(i)
                                  .tensorvec(j)
                                  .tensordatasize();
            memoryData.ptrData = reinterpret_cast<void *>(tensorPackageList
                                     ->tensorpackagevec(i)
                                     .tensorvec(j)
                                     .tensordataptr());
            if (memoryData.type == MxBase::MemoryData::MEMORY_HOST ||
                memoryData.type == MxBase::MemoryData::MEMORY_HOST_MALLOC ||
                memoryData.type == MxBase::MemoryData::MEMORY_HOST_NEW) {
                memoryData.deviceId = -1;
            }
            std::vector<uint32_t> outputShape = {};
            for (int k = 0; k < tensorPackageList->tensorpackagevec(i)
                                    .tensorvec(j)
                                    .tensorshape_size();
                 ++k) {
                outputShape.push_back(
                    (uint32_t)tensorPackageList->tensorpackagevec(i)
                        .tensorvec(j)
                        .tensorshape(k));
            }
            MxBase::TensorBase tmpTensor(
                memoryData, true, outputShape,
                (MxBase::TensorDataType)tensorPackageList->tensorpackagevec(0)
                    .tensorvec(j)
                    .tensordatatype());
            tensors.push_back(tmpTensor);
        }
    }
}

int main(int argc, char *argv[]) {
    LogInfo << "Usage: ./main --pipeline ../../data/config/Deepfm_ms.pipeline"
            << std::endl;

    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = google::GLOG_ERROR;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    LogInfo << "feat_ids :" << FLAGS_feat_ids;
    LogInfo << "feat_vals :" << FLAGS_feat_vals;
    LogInfo << "sample_num :" << FLAGS_sample_num;

    std::string feat_ids, feat_vals;
    std::string pipelinePath;

    // read pipeline config file
    std::string pipelineConfig;
    APP_ERROR ret = ReadFileToMem(FLAGS_pipeline, pipelineConfig);
    if (ret != APP_ERR_OK) {
        LogError << "Read pipeline file failed.";
        return APP_ERR_COMM_INIT_FAIL;
    }

    // init stream manager
    auto mxStreamManager = std::make_shared<MxStream::MxStreamManager>();
    ret = mxStreamManager->InitManager();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to init Stream manager, ret = " << ret << ".";
        return ret;
    }

    FLAGS_minloglevel = google::GLOG_ERROR;
    // create stream by pipeline config file
    ret = mxStreamManager->CreateMultipleStreams(pipelineConfig);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create Stream, ret = " << ret << ".";
        return ret;
    }

    timespec start_time, stop_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (int i = 0; i < FLAGS_sample_num; ++i) {
        if (!loadBinaryAsInput<int>(feat_ids, FLAGS_feat_ids, COL_PER_SAMPLE,
                                    i * COL_PER_SAMPLE, 0)) {
            LogError << "load text:" << FLAGS_feat_ids
                     << " to device input[0] error!!!";
            break;
        }

        if (!loadBinaryAsInput<float>(feat_vals, FLAGS_feat_vals,
                                      COL_PER_SAMPLE, i * COL_PER_SAMPLE, 1)) {
            LogError << "load text:" << FLAGS_feat_vals
                     << " to device input[0] error!!!";
            break;
        }

        // read image file and build stream input
        const char *dataPtr[2] = {feat_ids.c_str(), feat_vals.c_str()};
        int dataSize[2] = {sizeof(int) * COL_PER_SAMPLE,
                           sizeof(float) * COL_PER_SAMPLE};
        int dataShape[2] = {COL_PER_SAMPLE, COL_PER_SAMPLE};
        int32_t dataType[2] = {MxBase::TENSOR_DTYPE_INT32,
                               MxBase::TENSOR_DTYPE_FLOAT32};

        for (int j = 0; j < 2; ++j) {
            SendEachProtoBuf(j, dataPtr[j], dataSize[j], dataShape[j],
                             dataType[j], mxStreamManager, STREAM_NAME);
        }

        // get stream output
        std::vector<std::string> keyVec;
        keyVec.push_back("mxpi_tensorinfer0");
        std::vector<MxStream::MxstProtobufOut> output =
            mxStreamManager->GetProtobuf(STREAM_NAME, 0, keyVec);
        if (output.size() == 0) {
            LogError << "output size is 0";
            return APP_ERR_ACL_FAILURE;
        }
        if (output[0].errorCode != APP_ERR_OK) {
            LogError << "GetProtobuf error. errorCode=" << output[0].errorCode;
            return output[0].errorCode;
        }

        auto tensorPackageList =
            std::static_pointer_cast<MxTools::MxpiTensorPackageList>(
                output[0].messagePtr);
        std::vector<MxBase::TensorBase> tensors = {};
        GetTensors(tensorPackageList, tensors);

        void *tensorPtr = tensors[0].GetBuffer();
        float prediction = *reinterpret_cast<float *>(tensorPtr);
        if (tensors[0].GetSize() != 1) {
            LogError << "returned size should be 1.";
        }
        std::cout << prediction << std::endl;

        if (FLAGS_sample_num < 1000 || i % 1000 == 0) {
            LogInfo << "loading data index " << i;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &stop_time);
    double duration = (stop_time.tv_sec - start_time.tv_sec) +
                      (stop_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;

    LogError << "sec: " << duration;
    LogError << "fps: " << FLAGS_sample_num / duration;

    // destroy streams
    mxStreamManager->DestroyAllStreams();
}

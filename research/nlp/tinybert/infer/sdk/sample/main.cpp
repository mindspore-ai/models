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

#include <string>

#include "../../opensource/tokenizer.h"
#include "MxBase/Log/Log.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/Proto/MxpiDataTypeDeleter.h"

namespace {
const char* STREAM_NAME = "im_tinybert";
const int INT32_BYTELEN = 4;
const int BERT_INPUT_NUM = 3;
}  // namespace

DEFINE_string(input_file, "../../data/dataset/input_file.txt", "input_file.");
DEFINE_string(vocab_txt, "../../data/dataset/vocab.txt", "input vocab_txt.");
DEFINE_int32(max_seq_length, 128, "max_seq_length");
DEFINE_string(pipeline, "../../data/config/tinybert_ms.pipeline",
              "config file for this model.");
DEFINE_string(eval_labels_file, "../../data/dataset/eval_labels.txt",
              "eval labels file path.");

// This function is only for reading sample txt.
APP_ERROR SendEachProtobuf(
    int inPluginId, vector<int32_t> vec,
    shared_ptr<MxStream::MxStreamManager> &mxStreamManager) {
    auto dataSize = vec.size() * INT32_BYTELEN;
    auto dataPtr = &vec[0];

    MxBase::MemoryData memorySrc(dataPtr, dataSize,
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
    tensorVec->set_tensordatatype(MxBase::TENSOR_DTYPE_INT32);
    tensorVec->set_memtype(MxTools::MXPI_MEMORY_HOST_NEW);
    tensorVec->set_deviceid(0);
    tensorVec->add_tensorshape(1);
    tensorVec->add_tensorshape(vec.size());

    MxStream::MxstProtobufIn dataBuffer;
    ostringstream dataSource;
    dataSource << "appsrc" << inPluginId;
    dataBuffer.key = dataSource.str();
    dataBuffer.messagePtr =
        static_pointer_cast<google::protobuf::Message>(tensorPackageList);
    vector<MxStream::MxstProtobufIn> dataBufferVec;
    dataBufferVec.push_back(dataBuffer);
    ret = mxStreamManager->SendProtobuf(STREAM_NAME, inPluginId, dataBufferVec);
    return ret;
}

string ReadPipelineConfig() {
    ifstream file(FLAGS_pipeline.c_str(), ifstream::binary);
    if (!file) {
        LogError << FLAGS_pipeline << " file is not exists";
        return "";
    }
    file.seekg(0, ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);
    unique_ptr<char[]> data(new char[fileSize]);
    file.read(data.get(), fileSize);
    file.close();
    string pipelineConfig(data.get(), fileSize);
    return pipelineConfig;
}

vector<int> ReadEvalLabelTxt() {
    ifstream in(FLAGS_eval_labels_file.c_str());
    string line;
    vector<int> labels_vector;
    while (getline(in, line) && line.length() != 0) {
        if (line.empty()) {
            continue;
        }
        int label = atoi(line.c_str());
        labels_vector.push_back(label);
    }
    LogInfo << "eval labels size : " << labels_vector.size();
    return labels_vector;
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
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    string pipelineConfig = ReadPipelineConfig();
    if (pipelineConfig == "") {
        return APP_ERR_COMM_INIT_FAIL;
    }
    vector<int> eval_labels_vector = ReadEvalLabelTxt();
    if (eval_labels_vector.size() == 0) {
        LogError << "eval labels vector is empty !!!";
    }
    auto mxStreamManager = make_shared<MxStream::MxStreamManager>();
    APP_ERROR ret = mxStreamManager->InitManager();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to init Stream manager, ret = " << ret << ".";
        return ret;
    }
    ret = mxStreamManager->CreateMultipleStreams(pipelineConfig);

    if (ret != APP_ERR_OK) {
        LogError << "Failed to create Stream, ret = " << ret << ".";
        return ret;
    }

    const clock_t begin_time = clock();

    ifstream in(FLAGS_input_file.c_str());
    string line;
    BertTokenizer tokenizer;
    tokenizer.add_vocab(FLAGS_vocab_txt.c_str());

    int sample_index = 0;
    int correct_count = 0;
    while (getline(in, line) && line.length() != 0) {
        vector<int32_t> input_ids;
        vector<int32_t> input_segments;
        vector<int32_t> input_masks;
        tokenizer.encode(line, "", input_ids, input_masks, input_segments,
                         FLAGS_max_seq_length, "only_first");

        SendEachProtobuf(0, input_ids, mxStreamManager);
        SendEachProtobuf(1, input_segments, mxStreamManager);
        SendEachProtobuf(2, input_masks, mxStreamManager);
        vector<string> keyVec = {"mxpi_tensorinfer0"};

        vector<MxStream::MxstProtobufOut> output =
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
        vector<MxBase::TensorBase> tensors = {};
        GetTensors(tensorPackageList, tensors);
        void *tensorPtr = tensors[10].GetBuffer();
        uint32_t size = tensors[10].GetSize();
        float *ptr = reinterpret_cast<float *>(tensorPtr);
        int pred = ptr[0] > ptr[1] ? 0 : 1;
        if (pred == eval_labels_vector[sample_index]) {
            correct_count++;
        }
        if (sample_index % 100 == 0) {
            LogInfo << "loading data index " << sample_index;
        }
        sample_index++;
    }
    LogInfo << "performance summary";
    LogInfo << "#####################";
    LogInfo << "total samples: " <<
     sample_index;
    LogInfo << "accuracy: " << (static_cast<float>(correct_count)
                                 / static_cast<float>(sample_index));
    LogInfo << "cost time: " << (static_cast<float>(clock() - begin_time)
                                 / CLOCKS_PER_SEC);
    LogInfo << "sentences/s: "
            << sample_index / (static_cast<float>(clock() - begin_time)
             / CLOCKS_PER_SEC);
    // destroy streams
    mxStreamManager->DestroyAllStreams();
}

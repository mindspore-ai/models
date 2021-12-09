/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <bits/stdc++.h>
#include <string.h>
#include "MxStream/StreamManager/MxStreamManager.h"
#include "MxBase/Log/Log.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/Proto/MxpiDataTypeDeleter.h"


#define INT32_BYTELEN 4
APP_ERROR ReadFile(const std::string &filePath, MxStream::MxstDataInput* dataBuffer) {
    char c[PATH_MAX + 1] = {0x00};
    size_t count = filePath.copy(c, PATH_MAX + 1);
    if (count != filePath.length()) {
        LogError << "Failed to strcpy " << c;
        return APP_ERR_COMM_FAILURE;
    }

    char path[PATH_MAX + 1] = {0x00};
    if ((strlen(c) > PATH_MAX) || (realpath(c, path) == nullptr)) {
        LogError << "Failed to get image path(" << filePath << ").";
        return APP_ERR_COMM_NO_EXIST;
    }
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LogError << "Failed to open file";
        return APP_ERR_COMM_OPEN_FAIL;
    }

    fseek(fp, 0, SEEK_END);
    uint32_t fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (fileSize > 0) {
        dataBuffer->dataSize = fileSize;
        dataBuffer->dataPtr = new (std::nothrow) uint32_t[fileSize];
        if (dataBuffer->dataPtr == nullptr) {
            LogError << "allocate memory with \"new uint32_t\" failed.";
            fclose(fp);
            return APP_ERR_COMM_FAILURE;
        }
        fread(dataBuffer->dataPtr, 1, fileSize, fp);
        fclose(fp);
        return APP_ERR_OK;
    }
    fclose(fp);
    return APP_ERR_COMM_FAILURE;
}
std::string ReadPipelineConfig(const std::string &pipelineConfigPath) {
    std::ifstream file(pipelineConfigPath.c_str(), std::ifstream::binary);
    if (!file) {
        LogError << pipelineConfigPath << " file is not exists";
        return "";
    }
    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);
    std::unique_ptr<char[]> data(new char[fileSize]);
    file.read(data.get(), fileSize);
    file.close();
    std::string pipelineConfig(data.get(), fileSize);
    return pipelineConfig;
}

APP_ERROR SendEachProtobuf(std::string STREAM_NAME,
        int inPluginId,
        std::vector<int32_t> vec,
        const std::shared_ptr<MxStream::MxStreamManager> &mxStreamManager) {
    auto dataSize = vec.size() * INT32_BYTELEN;
    auto dataPtr = &vec[0];
    MxBase::MemoryData memorySrc(dataPtr, dataSize, MxBase::MemoryData::MEMORY_HOST_NEW);
    MxBase::MemoryData memoryDst(dataSize, MxBase::MemoryData::MEMORY_HOST_NEW);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, memorySrc);
    if (ret != APP_ERR_OK) {
        LogError << "Fail to malloc and copy host memory.";
        return ret;
    }

    auto tensorPackageList = std::shared_ptr<MxTools::MxpiTensorPackageList>(new MxTools::MxpiTensorPackageList,
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
    std::ostringstream dataSource;
    dataSource << "appsrc" << inPluginId;

    dataBuffer.key = dataSource.str();
    dataBuffer.messagePtr = std::static_pointer_cast<google::protobuf::Message>(tensorPackageList);
    std::vector<MxStream::MxstProtobufIn> dataBufferVec;
    dataBufferVec.push_back(dataBuffer);
    ret = mxStreamManager->SendProtobuf(STREAM_NAME, inPluginId, dataBufferVec);
    return ret;
}

void GetTensors(
    const std::shared_ptr<MxTools::MxpiTensorPackageList> &tensorPackageList,
    std::vector<MxBase::TensorBase>* tensors) {
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
            memoryData.ptrData = reinterpret_cast<void *>(tensorPackageList->tensorpackagevec(i)
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
            tensors->push_back(tmpTensor);
        }
    }
}

static float half_to_float(uint16_t h) {
    int16_t *ptr;
    int fs, fe, fm, rlt;

    ptr = reinterpret_cast<int16_t *>(&h);

    fs = ((*ptr) & 0x8000) << 16;

    fe = ((*ptr) & 0x7c00) >> 10;
    fe = fe + 0x70;
    fe = fe << 23;

    fm = ((*ptr) & 0x03ff) << 13;

    rlt = fs | fe | fm;
    // return *((float *)&rlt);
    return static_cast<float>(rlt);
}

void check(bool ret) {
    if (!ret) {
        LogError << "Fail to read.";
    }
}

void check_app_error(APP_ERROR ret) {
    if (ret != APP_ERR_OK) {
        LogError << "Failed to init Stream manager, ret = " << ret << ".";
    }
}

int main() {
    MxStream::MxstDataInput dataInput;

    std::string pipelineConfigPath = "../../data/config/bgcf.pipeline";
    std::string pipelineConfig = ReadPipelineConfig(pipelineConfigPath);
    if (pipelineConfig == "") {
        return APP_ERR_COMM_INIT_FAIL;
    }

    auto mxStreamManager = std::make_shared<MxStream::MxStreamManager>();
    APP_ERROR ret = mxStreamManager->InitManager();
    check_app_error(ret);
    ret = mxStreamManager->CreateMultipleStreams(pipelineConfig);
    check_app_error(ret);

    std::shared_ptr<MxTools::MxpiVisionList> objectList = std::make_shared<MxTools::MxpiVisionList>();
    MxStream::MxstProtobufIn dataBuffer;
    dataBuffer.key = "appsrc0";
    dataBuffer.messagePtr = std::static_pointer_cast<google::protobuf::Message>(objectList);
    std::vector<MxStream::MxstProtobufIn> dataBufferVec;
    dataBufferVec.push_back(dataBuffer);

    std::string streamName = "bgcf_gnn";
    int inPluginId = 0;
    std::vector<std::string> keyVec;
    keyVec.push_back("mxpi_tensorinfer0");

    int nums[7] = {7068, 3570, 3570, 282720, 141360, 142800, 71400};
    std::string names[7] = {"./dataset/users.txt", "./dataset/items.txt",
    "./dataset/neg_items.txt", "./dataset/u_test_neighs.txt",
    "./dataset/u_test_gnew_neighs.txt", "./dataset/i_test_neighs.txt",
    "./dataset/i_test_gnew_neighs.txt"};
    for (int i = 0; i < 7; i++) {
        if (freopen(names[i].c_str(), "r", stdin) != NULL) {
            std::vector<int32_t> vec(nums[i]);
            for (int j = 0; j < nums[i]; j++)
                std::cin >> vec[j];
            ret = SendEachProtobuf(streamName, i, vec, mxStreamManager);
            if (ret != APP_ERR_OK) {
                LogError << "Fail to malloc and copy host memory.";
                return ret;
            }
            fclose(stdin);
        }
    }

    std::vector<MxStream::MxstProtobufOut> output = mxStreamManager->GetProtobuf(streamName, inPluginId, keyVec);
    if (output.size() == 0) {
        LogError << "output size is 0";
        return APP_ERR_ACL_FAILURE;
    }
    if (output[0].errorCode != APP_ERR_OK) {
        LogError << "GetProtobuf error. errorCode=" << output[0].errorCode;
        return output[0].errorCode;
    }
    LogInfo << "errorCode=" << output[0].errorCode;
    LogInfo << "key=" << output[0].messageName;
    LogInfo << "value=" << output[0].messagePtr.get()->DebugString();

    auto tensorPackageList = std::static_pointer_cast<MxTools::MxpiTensorPackageList>(output[0].messagePtr);
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(tensorPackageList, &tensors);

    std::vector<std::vector<float>> user_rep(nums[0], std::vector<float>(192));
    std::vector<std::vector<float>> item_rep(nums[1], std::vector<float>(192));

    void *tensorPtr = tensors[0].GetBuffer();

    std::vector<unsigned int> sp = tensors[0].GetShape();
    std::cout << "GetShape_0=" << nums[0] << '*' << sp[1] << std::endl;

    uint16_t *ptr = reinterpret_cast<uint16_t *>(tensorPtr);

    check((freopen("../output/mxbase_user_rep.txt", "w", stdout) != NULL));
    for (int i = 0; i < nums[0]; i++) {
        for (int j = 0; j < 192; j++) {
            user_rep[i][j] = half_to_float(ptr[i * 192 + j]);
            std::cout << user_rep[i][j] << " ";
        }
        std::cout << std::endl;
    }

    check(freopen("/dev/tty", "w", stdout));

    void *tensorPtr1 = tensors[1].GetBuffer();

    std::vector<unsigned int> sp1 = tensors[1].GetShape();
    std::cout << "GetShape_1=" << nums[1] << '*' << sp1[1] << std::endl;

    uint16_t *ptr1 = reinterpret_cast<uint16_t *>(tensorPtr1);

    check(freopen("../output/mxbase_item_rep.txt", "w", stdout));

    for (int i = 0; i < nums[1]; i++) {
        for (int j = 0; j < 192; j++) {
            item_rep[i][j] = half_to_float(ptr1[i * 192 + j]);
            std::cout << item_rep[i][j] << " ";
        }
        std::cout << std::endl;
    }
    check((freopen("/dev/tty", "w", stdout) != NULL));

    std::cout << "user_rep[0][0~5] = " << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << user_rep[0][i] << " ";
    }
    std::cout << std::endl;

    std::cout << "item_rep[0][0~5] = " << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << item_rep[0][i] << " ";
    }
    std::cout << std::endl;

    mxStreamManager->DestroyAllStreams();
    return 0;
}


/*
 * Copyright (c) 2021.Huawei Technologies Co., Ltd. All rights reserved.
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

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "MxBase/Log/Log.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "acl/acl.h"

static inline const char *aclFormatToStr(int fmt) {
    switch (fmt) {
        case ACL_FORMAT_NCHW:
            return "NCHW";
        case ACL_FORMAT_NHWC:
            return "NHWC";
        case ACL_FORMAT_ND:
            return "ND";
        case ACL_FORMAT_NC1HWC0:
            return "NC1HWC0";
        case ACL_FORMAT_FRACTAL_Z:
            return "FRACTAL_Z";
        case ACL_FORMAT_FRACTAL_NZ:
            return "FRACTAL_NZ";
        default:
            return "UNDEFINED";
    }
}

static inline const char *aclDatatypeToStr(MxBase::TensorDataType ty) {
    switch (ty) {
        case ACL_FLOAT:
            return "float";
        case ACL_FLOAT16:
            return "float16";
        case ACL_INT8:
            return "int8";
        case ACL_INT32:
            return "int32";
        case ACL_UINT8:
            return "uint8";
        case ACL_INT16:
            return "int16";
        case ACL_UINT16:
            return "uint16";
        case ACL_UINT32:
            return "uint32";
        case ACL_INT64:
            return "int64";
        case ACL_UINT64:
            return "uint64";
        case ACL_DOUBLE:
            return "double";
        case ACL_BOOL:
            return "bool";
        default:
            return "undefined";
    }
}

class CBaseInfer {
 public:
    explicit CBaseInfer(uint32_t deviceId = 0) : m_nDeviceId(deviceId) {
        aclInit(nullptr);
        aclrtSetDevice(m_nDeviceId);
    }
    virtual ~CBaseInfer() {
        // DisposeDvpp();
        UnInit();
        aclrtResetDevice(m_nDeviceId);
        aclFinalize();
    }

    bool IsLoad() const { return m_bIsLoaded; }

    virtual bool Init(const std::string &omPath) {
        assert(!IsLoad());
        if (m_oInference.Init(omPath, m_oModelDesc) == APP_ERR_OK) {
            m_strModelPath = omPath;
            AllocateInputs(m_oMemoryInputs);
            allocate_outputs(m_oMemoryOutpus);
            m_bIsLoaded = true;
            return true;
        }

        return false;
    }

    void UnInit() {
        if (IsLoad()) {
            DisposeIOBuffers();
            m_oInference.DeInit();
            m_strModelPath.clear();
            m_bIsLoaded = false;
        }
    }

    void Dump() {
        LogInfo << "om file path:" << m_strModelPath;
        LogInfo << "model has " << m_oModelDesc.inputTensors.size()
                << " inputs, " << m_oModelDesc.outputTensors.size()
                << " outputs ...";

        size_t idx = 0;
        auto formats = m_oInference.GetInputFormat();
        auto dataTypes = m_oInference.GetInputDataType();
        for (auto &tensor : m_oModelDesc.inputTensors) {
            LogInfo << "input[" << idx
                    << "] dims: " << array_to_str(tensor.tensorDims)
                    << " size:" << tensor.tensorSize;

            LogInfo << "name: " << tensor.tensorName
                    << " format:" << aclFormatToStr(formats[idx])
                    << " dataType:" << aclDatatypeToStr(dataTypes[idx]);

            idx++;
        }

        idx = 0;
        formats = m_oInference.GetOutputFormat();
        dataTypes = m_oInference.GetOutputDataType();
        for (auto &tensor : m_oModelDesc.outputTensors) {
            LogInfo << "output[" << idx
                    << "] dims: " << array_to_str(tensor.tensorDims)
                    << " size:" << tensor.tensorSize;
            LogInfo << " name:" << tensor.tensorName
                    << " format:" << aclFormatToStr(formats[idx])
                    << " dataType:" << aclDatatypeToStr(dataTypes[idx]);
            idx++;
        }

        if (m_oModelDesc.dynamicBatch) {
            LogInfo << "dynamic batchSize: "
                    << array_to_str(m_oModelDesc.batchSizes);
        }
    }

    virtual MxBase::ObjectPostProcessorBase &GetPostProcessor() = 0;

    virtual void OutputResult(const std::vector<ObjDetectInfo> &) {}

    virtual APP_ERROR DoInference() {
        assert(IsLoad());

        MxBase::ObjectPostProcessorBase &postProcessor = GetPostProcessor();
        std::vector<MxBase::BaseTensor> inputTensors;
        std::vector<MxBase::BaseTensor> outputTensors;

        PrepareTensors(m_oMemoryInputs, m_oModelDesc.inputTensors,
                       inputTensors);
        PrepareTensors(m_oMemoryOutpus, m_oModelDesc.outputTensors,
                       outputTensors);

        APP_ERROR ret =
            m_oInference.ModelInference(inputTensors, outputTensors);

        if (ret == APP_ERR_OK) {
            auto featLayerData = std::vector<std::shared_ptr<void>>();
            std::vector<std::vector<MxBase::BaseTensor>> outputs = {
                outputTensors};

            ret = postProcessor.MemoryDataToHost(0, outputs, featLayerData);

            if (ret == APP_ERR_OK) {
                auto objInfos = std::vector<ObjDetectInfo>();
                MxBase::PostImageInfo unused;
                ret = postProcessor.Process(featLayerData, objInfos, false,
                                            unused);

                OutputResult(objInfos);
            }
        }
        return ret;
    }

    template <class Type>
    bool LoadBinaryAsInput(
        const std::string &filename,
        size_t struct_num = 1,     // number of <Type>s to read
        size_t offset = 0,         // pos to start reading, its unit is <Type>
        size_t input_index = 0) {  // index for the model inputs
        Type *data_record = new Type[struct_num];

        if (LoadBinaryFile<Type>(reinterpret_cast<char *>(data_record),
                                 filename, struct_num, offset) != APP_ERR_OK) {
            LogError << "load text:" << filename
                     << " to device input[0] error!!!";
            delete[] data_record;
            return false;
        }

        // LogInfo << "load text:" << filename << " to device successfully.";

        MxBase::MemoryData src_tmp(data_record,
                                   m_oMemoryInputs[input_index].size);
        MxBase::MemoryHelper::MxbsMemcpy(m_oMemoryInputs[input_index], src_tmp,
                                         src_tmp.size);

        delete[] data_record;
        return true;
    }

    template <class Type>
    int LoadBinaryFile(char *buffer, const std::string &filename,
                       int struct_num, size_t offset = 0) {
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

 protected:
    void DisposeIOBuffers() {
        for (auto &pos : m_oMemoryInputs) {
            MxBase::MemoryHelper::MxbsFree(pos);
        }
        m_oMemoryInputs.clear();
        for (auto &pos : m_oMemoryOutpus) {
            MxBase::MemoryHelper::MxbsFree(pos);
        }
        m_oMemoryOutpus.clear();
    }

    // prepare batch memory for inputs
    void AllocateInputs(std::vector<MxBase::MemoryData> &data) {
        data.resize(m_oModelDesc.inputTensors.size());
        size_t idx = 0;
        for (auto &mem : data) {
            mem.deviceId = m_nDeviceId;
            mem.type = MxBase::MemoryData::MEMORY_DEVICE;
            mem.size = m_oModelDesc.inputTensors[idx].tensorSize;
            MxBase::MemoryHelper::MxbsMalloc(mem);
            idx++;
        }
    }

    // prepare batch memory for outputs
    void allocate_outputs(std::vector<MxBase::MemoryData> &data) {
        data.resize(m_oModelDesc.outputTensors.size());
        size_t idx = 0;
        for (auto &mem : data) {
            mem.deviceId = m_nDeviceId;
            mem.type = MxBase::MemoryData::MEMORY_DEVICE;
            mem.size = m_oModelDesc.outputTensors[idx].tensorSize;
            MxBase::MemoryHelper::MxbsMalloc(mem);
            idx++;
        }
    }

    static MxBase::BaseTensor transform_tensor(const MxBase::MemoryData &pos,
                                               MxBase::TensorDesc &desc) {
        return MxBase::BaseTensor{
            pos.ptrData,
            std::vector<int>(desc.tensorDims.begin(), desc.tensorDims.end()),
            pos.size};
    }

    void PrepareTensors(std::vector<MxBase::MemoryData> &data,
                        std::vector<MxBase::TensorDesc> &tensorDesc,
                        std::vector<MxBase::BaseTensor> &tensor) {
        tensor.resize(data.size());
        std::transform(data.begin(), data.end(), tensorDesc.begin(),
                       tensor.begin(), transform_tensor);
    }

    APP_ERROR fetch_device_buffers(const std::vector<MxBase::MemoryData> &data,
                                   std::vector<std::shared_ptr<void>> &out) {
        for (auto &mem : data) {
            MxBase::MemoryData memoryDst(mem.size);

            APP_ERROR ret =
                MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, mem);
            if (ret != APP_ERR_OK) {
                LogError << GetError(ret)
                         << " Fail to copy device memory to host for "
                            "ModelPostProcessor.";
                return ret;
            }
            std::shared_ptr<void> buffer = nullptr;
            buffer.reset(memoryDst.ptrData, memoryDst.free);
            out.emplace_back(buffer);
        }
        return APP_ERR_OK;
    }

    // helper function to join array data into string
    template <class T>
    static std::string array_to_str(const std::vector<T> &vec) {
        std::ostringstream ostr;
        for (auto &pos : vec) ostr << pos << " ";
        return ostr.str();
    }

 protected:
    // om file path
    std::string m_strModelPath;
    // model description
    MxBase::ModelDesc m_oModelDesc;
    // MxBase inference module
    MxBase::ModelInferenceProcessor m_oInference;

    bool m_bIsLoaded = false;

    // device id used
    uint32_t m_nDeviceId = 0;
    // statically input & output buffers
    std::vector<MxBase::MemoryData> m_oMemoryInputs;
    std::vector<MxBase::MemoryData> m_oMemoryOutpus;
};

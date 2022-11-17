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

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "acl/acl.h"

class CBaseInfer {
 public:
    explicit CBaseInfer(uint32_t deviceId = 0) : m_nDeviceId(deviceId) {}
    virtual ~CBaseInfer() {}

    CBaseInfer(const CBaseInfer &) = delete;
    CBaseInfer &operator=(const CBaseInfer &) = delete;

    bool IsLoad() const { return m_bIsLoaded; }

    virtual bool Init(const std::string &omPath) {
        APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
        if (ret != APP_ERR_OK) {
            LogError << "Init devices failed, ret=" << ret << ".";
            return ret;
        }
        ret = MxBase::TensorContext::GetInstance()->SetContext(m_nDeviceId);
        if (ret != APP_ERR_OK) {
            LogError << "Set context failed, ret=" << ret << ".";
            return ret;
        }

        if (m_oInference.Init(omPath, m_oModelDesc) != APP_ERR_OK) {
            return false;
        }

        m_strModelPath = omPath;
        AllocateInputs(m_oMemoryInputs);
        allocate_outputs(m_oMemoryOutpus);
        m_bIsLoaded = true;
        return true;
    }

    virtual void UnInit() {
        if (IsLoad()) {
            DisposeIOBuffers();
            m_oInference.DeInit();
            m_strModelPath.clear();
            m_bIsLoaded = false;
            MxBase::DeviceManager::GetInstance()->DestroyDevices();
        }
    }

    void Dump() {
        LogInfo << "om file path:" << m_strModelPath;
        LogInfo << "model has " << m_oModelDesc.inputTensors.size()
                << " inputs, " << m_oModelDesc.outputTensors.size()
                << " outputs ...";

        size_t idx = 0;
        for (auto &tensor : m_oModelDesc.inputTensors) {
            LogInfo << "input[" << idx
                    << "] dims: " << array_to_str(tensor.tensorDims)
                    << " size:" << tensor.tensorSize;

            LogInfo << "name: " << tensor.tensorName
                    << " format:" << (m_oInference.GetInputFormat()[idx])
                    << " dataType:" << (m_oInference.GetInputDataType()[idx]);

            idx++;
        }

        idx = 0;
        for (auto &tensor : m_oModelDesc.outputTensors) {
            LogInfo << "output[" << idx
                    << "] dims: " << array_to_str(tensor.tensorDims)
                    << " size:" << tensor.tensorSize;
            LogInfo << " name:" << tensor.tensorName
                    << " format:" << (m_oInference.GetOutputFormat()[idx])
                    << " dataType:" << (m_oInference.GetOutputDataType()[idx]);
            idx++;
        }

        if (m_oModelDesc.dynamicBatch) {
            LogInfo << "dynamic batchSize: "
                    << array_to_str(m_oModelDesc.batchSizes);
        }
    }

    virtual MxBase::ObjectPostProcessorBase &get_post_processor() = 0;

    virtual void OutputResult(const std::vector<ObjDetectInfo> &) {}

    virtual APP_ERROR DoInference() {
        MxBase::ObjectPostProcessorBase &postProcessor = get_post_processor();
        std::vector<MxBase::BaseTensor> inputTensors;
        std::vector<MxBase::BaseTensor> outputTensors;

        prepare_tensors(m_oMemoryInputs, m_oModelDesc.inputTensors,
                        inputTensors);
        prepare_tensors(m_oMemoryOutpus, m_oModelDesc.outputTensors,
                        outputTensors);

        APP_ERROR ret =
            m_oInference.ModelInference(inputTensors, outputTensors);

        if (ret != APP_ERR_OK) {
            LogError << "Failed in ModelInference";
            return ret;
        }

        // auto metaDataPtr = std::make_shared<MxTools::MxpiObjectList>();
        auto featLayerData = std::vector<std::shared_ptr<void>>();
        std::vector<std::vector<MxBase::BaseTensor>> outputs = {outputTensors};

        ret = postProcessor.MemoryDataToHost(0, outputs, featLayerData);

        if (ret != APP_ERR_OK) {
            LogError << "Failed in MemoryDataToHost";
            return ret;
        }

        auto objInfos = std::vector<ObjDetectInfo>();
        MxBase::PostImageInfo unused;
        ret = postProcessor.Process(featLayerData, objInfos, false, unused);

        if (ret != APP_ERR_OK) {
            LogError << "Failed in postProcessor.Process()";
            return ret;
        }

        OutputResult(objInfos);

        return ret;
    }

    template <class Type>
    bool loadBinaryAsInput(
        const std::string &filename,
        size_t struct_num = 1,     // number of <Type>s to read
        size_t offset = 0,         // pos to start reading, its unit is <Type>
        size_t input_index = 0) {  // index for the model inputs
        Type *data_record = new (std::nothrow) Type[struct_num];
        if (!data_record) {
            LogError << "Failed to assign memory for data_record";
            return false;
        }

        if (LoadBinaryFile<Type>(data_record, filename, struct_num, offset) !=
            APP_ERR_OK) {
            LogError << "load text:" << filename
                     << " to device input[0] error!!!";
            delete[] data_record;
            return false;
        }

        MxBase::MemoryData src_tmp(data_record,
                                   m_oMemoryInputs[input_index].size);
        MxBase::MemoryHelper::MxbsMemcpy(m_oMemoryInputs[input_index], src_tmp,
                                         src_tmp.size);

        delete[] data_record;
        return true;
    }

    template <class Type>
    int LoadBinaryFile(Type *buffer, const std::string &filename,
                       int struct_num, size_t offset = 0) {
        std::ifstream rf(filename, std::ios::in | std::ios::binary);

        if (!rf) {
            LogError << "Cannot open file!";
            return -1;
        }

        if (offset > 0) {
            rf.seekg(sizeof(Type) * offset, rf.beg);
        }

        rf.read(reinterpret_cast<char *>(buffer), sizeof(Type) * struct_num);

        if (!rf) {
            LogError << "Failed when reading file";
            return -1;
        }

        rf.close();

        if (!rf.good()) {
            LogError << "Error occurred at reading time!";
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
                                               const MxBase::TensorDesc &desc) {
        return MxBase::BaseTensor{
            pos.ptrData,
            std::vector<int>(desc.tensorDims.begin(), desc.tensorDims.end()),
            pos.size};
    }

    void prepare_tensors(std::vector<MxBase::MemoryData> &data,
                         std::vector<MxBase::TensorDesc> &tensorDesc,
                         std::vector<MxBase::BaseTensor> &tensor) {
        tensor.reserve(data.size());
        std::transform(data.begin(), data.end(), tensorDesc.begin(),
                       tensor.begin(), transform_tensor);
    }

    APP_ERROR fetch_device_buffers(std::vector<MxBase::MemoryData> &data,
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
        for (auto &pos : vec) {
            ostr << pos << " ";
        }
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

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
#include <algorithm>
#include "MxBaseInfer.h"

CBaseInfer::CBaseInfer(uint32_t deviceId) : m_nDeviceId(deviceId) {
    aclInit(nullptr);
    aclrtSetDevice(m_nDeviceId);
}

CBaseInfer::~CBaseInfer() {
    aclrtResetDevice(m_nDeviceId);
    aclFinalize();
}

bool CBaseInfer::Init(const std::string &omPath) {
    assert(!IsLoad());
    if (m_oInference.Init(omPath, m_oModelDesc) == APP_ERR_OK) {
        m_strModelPath = omPath;
        AllocateInputs(m_oMemoryInputs);
        AllocateOutputs(m_oMemoryOutpus);
        m_bIsLoaded = true;
        return true;
    }
    return false;
}

void CBaseInfer::UnInit() {
    if (IsLoad()) {
        DisposeIOBuffers();
        m_oInference.DeInit();
        m_strModelPath.clear();
        m_bIsLoaded = false;
    }
}

void CBaseInfer::Dump() {
    LogInfo << "om file path:" << m_strModelPath;
    LogInfo << "model has " << m_oModelDesc.inputTensors.size() << " inputs, "
            << m_oModelDesc.outputTensors.size() << " outputs ...";

    size_t idx = 0;
    for (auto &tensor : m_oModelDesc.inputTensors) {
        LogInfo << "input[" << idx << "] dims: " << Array2Str(tensor.tensorDims)
                << " size:" << tensor.tensorSize;

        LogInfo << " name: " << tensor.tensorName
                << " format:" << (m_oInference.GetInputFormat()[idx])
                << " dataType:" << (m_oInference.GetInputDataType()[idx]);

        idx++;
    }

    idx = 0;
    for (auto &tensor : m_oModelDesc.outputTensors) {
        LogInfo << " output[" << idx
                << "] dims: " << Array2Str(tensor.tensorDims)
                << " size:" << tensor.tensorSize;
        LogInfo << " name:" << tensor.tensorName
                << " format:" << (m_oInference.GetOutputFormat()[idx])
                << " dataType:" << (m_oInference.GetOutputDataType()[idx]);
        idx++;
    }

    if (m_oModelDesc.dynamicBatch) {
        LogInfo << "dynamic batchSize: " << Array2Str(m_oModelDesc.batchSizes);
    }
}

APP_ERROR CBaseInfer::DoInference() {
    assert(IsLoad());

    MxBase::ObjectPostProcessorBase &postProcessor = GetPostProcessor();
    std::vector<MxBase::BaseTensor> inputTensors;
    std::vector<MxBase::BaseTensor> outputTensors;

    PrepareTensors(m_oMemoryInputs, m_oModelDesc.inputTensors, inputTensors);
    PrepareTensors(m_oMemoryOutpus, m_oModelDesc.outputTensors, outputTensors);

    APP_ERROR ret = m_oInference.ModelInference(inputTensors, outputTensors);

    if (ret == APP_ERR_OK) {
        auto featLayerData = std::vector<std::shared_ptr<void>>();
        std::vector<std::vector<MxBase::BaseTensor>> outputs = {outputTensors};

        ret = postProcessor.MemoryDataToHost(0, outputs, featLayerData);

        if (ret == APP_ERR_OK) {
            std::vector<ObjDetectInfo> objInfos;
            MxBase::PostImageInfo unused;
            ret = postProcessor.Process(featLayerData, objInfos, false, unused);
            this->predict_vector.push_back(objInfos[0].classId);
        }
    }
    return ret;
}

bool CBaseInfer::LoadVectorAsInput(
    std::vector<int> &vec,
    size_t input_index) {  // index for the model inputs
    MxBase::MemoryData src_tmp(&vec[0], m_oMemoryInputs[input_index].size);
    MxBase::MemoryHelper::MxbsMemcpy(m_oMemoryInputs[input_index], src_tmp,
                                     src_tmp.size);
    return true;
}

void CBaseInfer::DisposeIOBuffers() {
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
void CBaseInfer::AllocateInputs(std::vector<MxBase::MemoryData> &data) {
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
void CBaseInfer::AllocateOutputs(std::vector<MxBase::MemoryData> &data) {
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

void CBaseInfer::PrepareTensors(std::vector<MxBase::MemoryData> &data,
                                std::vector<MxBase::TensorDesc> &tensorDesc,
                                std::vector<MxBase::BaseTensor> &tensor) {
    tensor.resize(data.size());
    std::transform(data.begin(), data.end(), tensorDesc.begin(), tensor.begin(),
                   transform_tensor);
}

APP_ERROR CBaseInfer::FetchDeviceBuffers(
    std::vector<MxBase::MemoryData> &data,
    std::vector<std::shared_ptr<void>> &out) {
    for (auto &mem : data) {
        MxBase::MemoryData memoryDst(mem.size);

        APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDst, mem);
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

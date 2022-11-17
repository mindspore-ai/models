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
#include <memory>
#include <string>
#include <vector>

#include "MxBase/Log/Log.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/ModelPostProcessors/ModelPostProcessorBase/ObjectPostProcessorBase.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "acl/acl.h"

class CBaseInfer {
 public:
    explicit CBaseInfer(uint32_t deviceId = 0);

    virtual ~CBaseInfer();

    bool IsLoad() const { return m_bIsLoaded; }

    virtual bool Init(const std::string &omPath);

    virtual void UnInit();

    void Dump();

    virtual MxBase::ObjectPostProcessorBase &GetPostProcessor() = 0;
    virtual void OutputResult(const std::vector<ObjDetectInfo> &) {}

    virtual APP_ERROR DoInference();

    bool LoadVectorAsInput(std::vector<int> &vec, size_t input_index = 0);

 protected:
    void DisposeIOBuffers();

    // prepare batch memory for inputs
    void AllocateInputs(std::vector<MxBase::MemoryData> &data);

    // prepare batch memory for outputs
    void AllocateOutputs(std::vector<MxBase::MemoryData> &data);

    void PrepareTensors(std::vector<MxBase::MemoryData> &data,
                        std::vector<MxBase::TensorDesc> &tensorDesc,
                        std::vector<MxBase::BaseTensor> &tensor);

    APP_ERROR FetchDeviceBuffers(std::vector<MxBase::MemoryData> &data,
                                 std::vector<std::shared_ptr<void>> &out);

    // helper function to join array data into string
    template <class T>
    static std::string Array2Str(const std::vector<T> &vec) {
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
    // device id used
    bool m_bIsLoaded = false;
    uint32_t m_nDeviceId = 0;
    // statically input & output buffers
    std::vector<MxBase::MemoryData> m_oMemoryInputs;
    std::vector<MxBase::MemoryData> m_oMemoryOutpus;

 public:
    std::vector<int> predict_vector;
};

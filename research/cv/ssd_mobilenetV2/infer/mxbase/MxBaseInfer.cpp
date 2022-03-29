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

#include "infer/mxbase/MxBaseInfer.h"

#include <assert.h>

#include <vector>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "acl/acl.h"
#include "infer/mxbase/MxImage.h"
#include "opencv2/opencv.hpp"

namespace sdk_infer {
namespace mxbase_infer {

MxBaseInfer::MxBaseInfer(uint32_t deviceId)
    : m_deviceId(deviceId), m_is_init(false) {}

MxBaseInfer::~MxBaseInfer() { UnInit(); }

bool MxBaseInfer::IsInit() const { return m_is_init; }

bool MxBaseInfer::Init(const std::string &om_path) {
    BeforeInit();
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed,ret=" << ret << ".";
        return false;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(m_deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "SetContext failed,ret=" << ret << ".";
        return false;
    }
    if (IsInit()) {
        LogError << "Init fail,is init";
        return false;
    }
    ret = m_model_processor.Init(om_path, m_model_desc);
    if (ret != APP_ERR_OK) {
        LogError << "m_model_processor init error,ret=" << ret << ".";
        return false;
    }

    m_model_path = om_path;
    m_is_init = true;

    if (!AfterInit()) {
        LogError << "AfterInit fail.";
        return false;
    }

    return ret == APP_ERR_OK;
}

bool MxBaseInfer::UnInit() {
    if (IsInit()) {
        m_model_processor.DeInit();
        m_model_path.clear();
        m_model_desc = MxBase::ModelDesc();
        MxBase::DeviceManager::GetInstance()->DestroyDevices();
        m_is_init = false;
    }
    return true;
}

void MxBaseInfer::GetWidthAndHeightFromModel(uint32_t *w, uint32_t *h,
                                             size_t idx) {
    assert(m_model_desc.inputTensors[idx].tensorDims.size() == 4);

    aclFormat input_format = (aclFormat)m_model_processor.GetInputFormat()[idx];

    if (input_format == ACL_FORMAT_NCHW) {
        *w = m_model_desc.inputTensors[idx].tensorDims[3];
        *h = m_model_desc.inputTensors[idx].tensorDims[2];
    } else if (input_format == ACL_FORMAT_NHWC) {
        *w = m_model_desc.inputTensors[idx].tensorDims[2];
        *h = m_model_desc.inputTensors[idx].tensorDims[1];
    } else {
        assert(0);
    }
}

bool MxBaseInfer::Inference(std::vector<MxBase::TensorBase> *input_tensors,
                            std::vector<MxBase::TensorBase> *output_tensors) {
    if (!m_is_init) {
        LogError << "is un init.";
        return false;
    }

    auto dtypes = m_model_processor.GetOutputDataType();
    for (size_t i = 0; i < m_model_desc.outputTensors.size(); i++) {
        std::vector<uint32_t> shape{
            m_model_desc.outputTensors[i].tensorDims.begin(),
            m_model_desc.outputTensors[i].tensorDims.end()};
        output_tensors->emplace_back(
            shape, dtypes[i], MxBase::MemoryData::MEMORY_DEVICE, m_deviceId);
        APP_ERROR ret =
            MxBase::TensorBase::TensorBaseMalloc(output_tensors->back());
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed,ret=" << ret << ".";
            return false;
        }
    }

    MxBase::DynamicInfo dynamic_info = {};
    dynamic_info.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = m_model_processor.ModelInference(
        *input_tensors, *output_tensors, dynamic_info);

    if (ret != APP_ERR_OK) {
        LogInfo << "ModelInference error ret=" << GetAppErrCodeInfo(ret);
    }
    PostProcess(*output_tensors);
    return ret == APP_ERR_OK;
}

}  // namespace mxbase_infer
}  // namespace sdk_infer

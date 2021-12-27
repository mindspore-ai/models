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

#include "infer/mxbase/MxUtil.h"

#include "MxBase/Log/Log.h"

namespace sdk_infer {
namespace mxbase_infer {

std::string MxUtil::aclFormatToStr(aclFormat fmt) {
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

std::string MxUtil::tensorDatatypeToStr(MxBase::TensorDataType ty) {
    return MxBase::GetTensorDataTypeDesc(ty);
}

void MxUtil::LogModelDesc(
    const MxBase::ModelInferenceProcessor &model_inference_process,
    const MxBase::ModelDesc &desc) {
    LogInfo << "model has " << desc.inputTensors.size() << " inputs, "
            << desc.outputTensors.size() << " outputs ...";
    size_t idx = 0;
    auto formats = model_inference_process.GetInputFormat();
    auto dataTypes = model_inference_process.GetInputDataType();
    for (auto &tensor : desc.inputTensors) {
        LogInfo << "input[" << idx << "] dims: " << Array2Str(tensor.tensorDims)
                << " size:" << tensor.tensorSize;
        LogInfo << "name: " << tensor.tensorName
                << " format:" << aclFormatToStr((aclFormat)formats[idx])
                << " dataType:" << tensorDatatypeToStr(dataTypes[idx]);
        idx++;
    }
    idx = 0;
    formats = model_inference_process.GetOutputFormat();
    dataTypes = model_inference_process.GetOutputDataType();
    for (auto &tensor : desc.outputTensors) {
        LogInfo << "output[" << idx
                << "] dims: " << Array2Str(tensor.tensorDims)
                << " size:" << tensor.tensorSize;
        LogInfo << " name:" << tensor.tensorName
                << " format:" << aclFormatToStr((aclFormat)formats[idx])
                << " dataType:" << tensorDatatypeToStr(dataTypes[idx]);
        idx++;
    }
    if (desc.dynamicBatch) {
        LogInfo << "dynamic batchSize: " << Array2Str(desc.batchSizes);
    }
}

}  // namespace mxbase_infer
}  // namespace sdk_infer

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

#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "acl/acl.h"

namespace sdk_infer {
namespace mxbase_infer {

class MxUtil {
 public:
    std::string aclFormatToStr(aclFormat fmt);

    std::string tensorDatatypeToStr(MxBase::TensorDataType ty);

    void LogModelDesc(
        const MxBase::ModelInferenceProcessor &model_inference_process,
        const MxBase::ModelDesc &desc);

    template <typename T>
    std::string Array2Str(const std::vector<T> &vec) {
        std::ostringstream ostr;
        for (auto &elem : vec) {
            ostr << elem << " ";
        }
        return ostr.str();
    }
};

}  // namespace mxbase_infer
}  // namespace sdk_infer

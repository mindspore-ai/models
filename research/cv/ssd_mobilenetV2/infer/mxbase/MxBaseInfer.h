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

#include <cstdint>
#include <string>
#include <vector>

#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "opencv2/opencv.hpp"

namespace sdk_infer {
namespace mxbase_infer {

class MxBaseInfer {
 public:
    explicit MxBaseInfer(uint32_t deviceId);
    ~MxBaseInfer();

    bool IsInit() const;

    virtual bool BeforeInit() { return true; }
    bool Init(const std::string &om_path);
    virtual bool AfterInit() { return true; }
    bool UnInit();

    virtual void PostProcess(std::vector<MxBase::TensorBase> &) = 0;
    bool Inference(std::vector<MxBase::TensorBase> *input_tensors,
                   std::vector<MxBase::TensorBase> *output_tensors);

    // 从磁盘中将图像加载到模型的内存中
    virtual bool LoadImageToModel(const std::string &file,
                                  MxBase::TensorBase *tensor,
                                  size_t image_idx) = 0;

 protected:
    // 从模型中获取图像的期望宽度和高度
    void GetWidthAndHeightFromModel(uint32_t *w, uint32_t *h, size_t model_idx);

 protected:
    MxBase::ModelInferenceProcessor m_model_processor;
    MxBase::ModelDesc m_model_desc;
    uint32_t m_deviceId;
    bool m_is_init;

    std::string m_model_path;
};

}  // namespace mxbase_infer
}  // namespace sdk_infer

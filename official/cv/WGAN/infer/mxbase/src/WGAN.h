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
#ifndef WGAN_H
#define WGAN_H

#include <string>
#include <vector>
#include <memory>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "MxBase/DeviceManager/DeviceManager.h"

struct InitParam{
    uint32_t deviceId;
    std::string modelPath;
};

struct model_info{
    uint32_t noise_length;
    uint32_t nimages;
    uint32_t image_size;
};

class WGAN{
 public:
    APP_ERROR Generate_input_Tensor(const model_info &modelInfo, std::vector<MxBase::TensorBase> *input);
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &infer_outputs,
                           std::vector<float> *processed_result);
    APP_ERROR Process(const model_info &modelInfo, const std::string &resultPath);
    APP_ERROR WriteResult(const std::string &fileName, std::vector<float> *output_img_data);

 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
};

#endif  // WGAN_H

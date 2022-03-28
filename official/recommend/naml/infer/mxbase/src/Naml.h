/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MXBASE_NAML_H
#define MXBASE_NAML_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string newsmodelPath;
    std::string usermodelPath;
    std::string newsDataPath;
    std::string userDataPath;
    std::string evalDataPath;
};

class Naml {
 public:
    APP_ERROR process(const std::vector < std::string > & datapaths,
    const InitParam & initparam);
    APP_ERROR inference(const std::vector < MxBase::TensorBase > & inputs,
    std::vector < MxBase::TensorBase > * outputs,
    std::shared_ptr < MxBase::ModelInferenceProcessor > & model_,
    MxBase::ModelDesc desc);
    APP_ERROR input_news_tensor(std::vector < MxBase::TensorBase > * inputs,
    uint8_t index, uint32_t * data,
    uint32_t tensor_size);
    APP_ERROR input_user_tensor(std::vector < MxBase::TensorBase > * inputs,
    uint8_t index,
    std::vector < std::vector < float_t >> & userdata);
    APP_ERROR de_init();
    void calcAUC(std::vector < float > & vec_auc, std::vector < float > & predResult,
    std::vector < uint32_t > & labels);
    APP_ERROR readfile(const std::string & filepath,
    std::vector < std::vector < std::string >> & datastr);
    APP_ERROR read_news_inputs(std::vector < std::string > & datavec,
    std::vector < MxBase::TensorBase > * inputs);
    APP_ERROR read2arr(std::string & datastr, std::uint32_t * arr);
    APP_ERROR pred_process(std::vector < std::vector < std::string >> & eval_input,
    std::map < uint32_t, std::vector < float_t >> & news_ret_map,
    std::map < uint32_t, std::vector < float_t >> & user_ret_map);
    APP_ERROR news_process(std::vector < std::vector < std::string >> & news_input,
    std::map < uint32_t, std::vector < float_t >> & news_ret_map);
    APP_ERROR user_process(std::vector < std::vector < std::string >> & user_input,
    std::map < uint32_t, std::vector < float_t >> & user_ret_map,
    std::map < uint32_t, std::vector < float_t >> & news_ret_map);
    APP_ERROR read_user_inputs(std::vector < std::string > & datavec,
    std::map < uint32_t, std::vector < float_t >> & news_ret_map,
    std::vector < MxBase::TensorBase > * inputs);
    APP_ERROR post_process(std::vector < MxBase::TensorBase > * outputs,
    std::map < uint32_t, std::vector < float_t >> * ret_map,
    uint32_t index);
    APP_ERROR init(const InitParam & initParam);

 private:
    std::shared_ptr < MxBase::ModelInferenceProcessor > news_model_;
    std::shared_ptr < MxBase::ModelInferenceProcessor > user_model_;
    MxBase::ModelDesc news_modelDesc_ = {
    };
    MxBase::ModelDesc user_modelDesc_ = {
    };
    uint32_t deviceId_ = 0;
};
#endif

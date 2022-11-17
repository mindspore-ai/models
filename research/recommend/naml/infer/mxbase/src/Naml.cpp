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

#include "Naml.h"

#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR Naml::init(const InitParam & initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }

    news_model_ = std::make_shared < MxBase::ModelInferenceProcessor >();
    ret = news_model_->Init(initParam.newsmodelPath, news_modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "news_model_ init failed, ret=" << ret << ".";
        return ret;
    }

    user_model_ = std::make_shared < MxBase::ModelInferenceProcessor >();
    ret = user_model_->Init(initParam.usermodelPath, user_modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "user_model_ init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Naml::de_init() {
    news_model_->DeInit();
    user_model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Naml::process(const std::vector < std::string > & datapaths,
const InitParam & initParam) {
    APP_ERROR ret = init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Naml init newsmodel failed, ret=" << ret << ".";
        return ret;
    }
    std::vector < std::vector < std::string >> news_input;
    std::string input_news_path = datapaths[0];
    std::map < uint32_t, std::vector < float_t >> news_ret_map;
    ret = readfile(input_news_path, news_input);
    if (ret != APP_ERR_OK) {
        LogError << "read datafile failed, ret=" << ret << ".";
        return ret;
    }
    ret = news_process(news_input, news_ret_map);
    if (ret != APP_ERR_OK) {
        LogError << "news_process failed, ret=" << ret << ".";
        return ret;
    }

    std::vector < std::vector < std::string >> user_input;
    std::string input_user_path = datapaths[1];
    std::map < uint32_t, std::vector < float_t >> user_ret_map;
    ret = readfile(input_user_path, user_input);
    if (ret != APP_ERR_OK) {
        LogError << "read datafile failed, ret=" << ret << ".";
        return ret;
    }

    ret = user_process(user_input, user_ret_map, news_ret_map);
    if (ret != APP_ERR_OK) {
        LogError << "user_process failed, ret=" << ret << ".";
        return ret;
    }

    std::vector < std::vector < std::string >> eval_input;
    std::string input_eval_path = datapaths[2];

    ret = readfile(input_eval_path, eval_input);
    if (ret != APP_ERR_OK) {
        LogError << "read datafile failed, ret=" << ret << ".";
        return ret;
    }
    ret = pred_process(eval_input, news_ret_map, user_ret_map);
    if (ret != APP_ERR_OK) {
        LogError << "pred_process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

void string2vector(std::vector < uint32_t > & vec, std::string str) {
    std::istringstream sstr(str);
    while (sstr.good()) {
        uint32_t num = 0;
        sstr >> num;
        vec.push_back(num);
    }
}

APP_ERROR Naml::pred_process(std::vector < std::vector < std::string >> & eval_input,
std::map < uint32_t, std::vector < float_t >> & news_ret_map,
std::map < uint32_t, std::vector < float_t >> & user_ret_map) {
    std::vector < float > vec_AUC;
    for (size_t i = 0; i < eval_input.size() - 1; i++) {
        std::vector < std::vector < float >> newsCandidate;
        std::vector < uint32_t > candidateNewsIds;
        string2vector(candidateNewsIds, eval_input[i][1]);
        std::uint32_t candidateUid = 0;
        std::istringstream uid(eval_input[i][0]);
        uid >> candidateUid;
        std::vector < float > predResult;
        for (size_t j = 0; j < candidateNewsIds.size(); j++) {
            auto it = news_ret_map.find(candidateNewsIds[j]);
            if (it != news_ret_map.end()) {
                newsCandidate.push_back(news_ret_map[candidateNewsIds[j]]);
            }
        }
        for (size_t j = 0; j < newsCandidate.size(); j++) {
            float dotMulResult = 0;
            for (int k = 0; k < 400; ++k) {
                auto it = user_ret_map.find(candidateUid);
                if (it != user_ret_map.end()) {
                    dotMulResult += newsCandidate[j][k] * user_ret_map[candidateUid][k];
                }
            }
            predResult.push_back(dotMulResult);
        }
        if (predResult.size() > 0) {
            std::vector < uint32_t > labels;
            string2vector(labels, eval_input[i][2]);
            calcAUC(vec_AUC, predResult, labels);
            LogInfo << "The pred processing ：" << i << " / " << eval_input.size() - 1
            << std::endl;
        }
    }
    LogInfo << "The pred processing ：" << eval_input.size() - 1 << " / "
    << eval_input.size() - 1 << std::endl;
    float ans = 0;
    for (size_t i = 0; i < vec_AUC.size(); i++) {
        ans += vec_AUC[i];
    }
    ans = ans / vec_AUC.size();
    LogInfo << "The AUC is ：" << ans << std::endl;
    return APP_ERR_OK;
}

void Naml::calcAUC(std::vector < float > & vec_auc, std::vector < float > & predResult,
std::vector < uint32_t > & labels) {
    int N = 0, P = 0;
    std::vector < float > neg_prob;
    std::vector < float > pos_prob;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] == 1) {
            P += 1;
            pos_prob.push_back(predResult[i]);
        } else {
            N += 1;
            neg_prob.push_back(predResult[i]);
        }
    }
    float count = 0;
    for (size_t i = 0; i < pos_prob.size(); i++) {
        for (size_t j = 0; j < neg_prob.size(); j++) {
            if (pos_prob[i] > neg_prob[j]) {
                count += 1;
            } else if (pos_prob[i] == neg_prob[j]) {
                count += 0.5;
            }
        }
    }
    vec_auc.push_back(count / (N * P));
}

APP_ERROR Naml::news_process(std::vector < std::vector < std::string >> & news_input,
std::map < uint32_t, std::vector < float_t >> & news_ret_map) {
    for (size_t i = 0; i < news_input.size() - 1; i++) {
        std::vector < MxBase::TensorBase > inputs = {
        };
        std::vector < MxBase::TensorBase > outputs = {
        };
        APP_ERROR ret = read_news_inputs(news_input[i], &inputs);
        if (ret != APP_ERR_OK) {
            LogError << "get inputs failed, ret=" << ret << ".";
            return ret;
        }
        ret = inference(inputs, &outputs, news_model_, news_modelDesc_);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        std::uint32_t nid = 0;
        std::istringstream sstream(news_input[i][0]);
        sstream >> nid;
        ret = post_process(&outputs, &news_ret_map, nid);
        if (ret != APP_ERR_OK) {
            LogError << "post_process failed, ret=" << ret << ".";
            return ret;
        }
        LogInfo << "The news model is processing ：" << i << " / "
        << news_input.size() - 1 << std::endl;
    }
    LogInfo << "The news model completes the task：" << news_input.size() - 1
    << " / " << news_input.size() - 1 << std::endl;
    return APP_ERR_OK;
}

APP_ERROR Naml::user_process(std::vector < std::vector < std::string >> & user_input,
std::map < uint32_t, std::vector < float_t >> & user_ret_map,
std::map < uint32_t, std::vector < float_t >> & news_ret_map) {
    for (size_t i = 0; i < user_input.size() - 1; i++) {
        std::vector < MxBase::TensorBase > inputs = {
        };
        std::vector < MxBase::TensorBase > outputs = {
        };
        APP_ERROR ret = read_user_inputs(user_input[i], news_ret_map, &inputs);
        if (ret != APP_ERR_OK) {
            LogError << "get inputs failed, ret=" << ret << ".";
            return ret;
        }
        ret = inference(inputs, &outputs, user_model_, user_modelDesc_);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        std::uint32_t uid = 0;
        std::istringstream sstream(user_input[i][0]);
        sstream >> uid;
        ret = post_process(&outputs, &user_ret_map, uid);
        if (ret != APP_ERR_OK) {
            LogError << "post_process failed, ret=" << ret << ".";
            return ret;
        }
        LogInfo << "The user model is processing ：" << i << " / "
        << user_input.size() - 1 << std::endl;
    }
    LogInfo << "The user model completes the task：" << user_input.size() - 1
    << " / " << user_input.size() - 1 << std::endl;
    return APP_ERR_OK;
}

APP_ERROR Naml::readfile(const std::string & filepath,
std::vector < std::vector < std::string >> & datastr) {
    std::ifstream infile;
    infile.open(filepath, std::ios_base::in);
    if (infile.fail()) {
        LogError << "Failed to open file: " << filepath << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::string linestr;
    while (infile.good()) {
        std::getline(infile, linestr);
        std::istringstream sstream(linestr);
        std::vector < std::string > vecstr;
        std::string str;
        while (std::getline(sstream, str, ',')) {
            vecstr.push_back(str);
        }
        datastr.push_back(vecstr);
    }
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR Naml::post_process(std::vector < MxBase::TensorBase > * outputs,
std::map < uint32_t, std::vector < float_t >> * ret_map,
uint32_t index) {
    MxBase::TensorBase & tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // check tensor is available
    auto outputShape = tensor.GetShape();
    uint32_t length = outputShape[0];
    uint32_t classNum = outputShape[1];
    // LogInfo << "output shape is: " << outputShape[0] << " " << outputShape[1]
    // << std::endl;
    void * data = tensor.GetBuffer();
    for (uint32_t i = 0; i < length; i++) {
        std::vector < float_t > result = {
        };
        for (uint32_t j = 0; j < classNum; j++) {
            float_t value = *(reinterpret_cast < float_t * > (data) + i * classNum + j);
            result.push_back(value);
        }
        ret_map->insert(std::pair < uint32_t, std::vector < float_t >>(index, result));
    }
    return APP_ERR_OK;
}

APP_ERROR Naml::inference(const std::vector < MxBase::TensorBase > & inputs,
std::vector < MxBase::TensorBase > * outputs,
std::shared_ptr < MxBase::ModelInferenceProcessor > & model_,
MxBase::ModelDesc desc) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < desc.outputTensors.size(); ++i) {
        std::vector < uint32_t > shape = {
        };
        for (size_t j = 0; j < desc.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)desc.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i],
        MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
        deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {
    };
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Naml::read_user_inputs(std::vector < std::string > & datavec,
std::map < uint32_t, std::vector < float_t >> & news_ret_map,
std::vector < MxBase::TensorBase > * inputs) {
    uint32_t history[50] = {
        0
    };
    read2arr(datavec[1], history);
    std::vector < std::vector < float_t >> userdata;
    for (size_t i = 0; i < 50; i++) {
        auto it = news_ret_map.find(history[i]);
        if (it != news_ret_map.end()) {
            userdata.push_back(news_ret_map[history[i]]);
        }
    }
    input_user_tensor(inputs, 0, userdata);
    return APP_ERR_OK;
}

APP_ERROR Naml::read_news_inputs(std::vector < std::string > & datavec,
std::vector < MxBase::TensorBase > * inputs) {
    int length[] = {
        1, 1, 16, 48
    };
    for (int i = 0; i < 4; i++) {
        uint32_t data[length[i]] = {
            0
        };
        read2arr(datavec[i + 1], data);
        APP_ERROR ret = input_news_tensor(inputs, i, data, length[i]);
        if (ret != APP_ERR_OK) {
            LogError << "input array failed.";
            return ret;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR Naml::read2arr(std::string & datastr, std::uint32_t * arr) {
    std::istringstream str(datastr);
    while (str.good()) {
        str >> *arr;
        arr++;
    }
    return APP_ERR_OK;
}

APP_ERROR Naml::input_user_tensor(std::vector < MxBase::TensorBase > * inputs,
uint8_t index,
std::vector < std::vector < float_t >> & userdata) {
    float_t data[50][400] = {
        0
    };
    for (size_t i = 0; i < userdata.size(); i++) {
        for (size_t j = 0; j < 400; j++) {
            data[i][j] = userdata[i][j];
        }
    }
    const uint32_t dataSize = user_modelDesc_.inputTensors[index].tensorSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE,
    deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast < void * > (data), dataSize,
    MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret =
    MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }
    std::vector < uint32_t > shape = {
        1, 50, 400
    };
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape,
    MxBase::TENSOR_DTYPE_FLOAT32));
    return APP_ERR_OK;
}

APP_ERROR Naml::input_news_tensor(std::vector < MxBase::TensorBase > * inputs,
uint8_t index, uint32_t * data,
uint32_t tensor_size) {
    const uint32_t dataSize = news_modelDesc_.inputTensors[index].tensorSize;

    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE,
    deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast < void * > (data), dataSize,
    MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret =
    MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }
    std::vector < uint32_t > shape = {
        1, tensor_size
    };
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape,
    MxBase::TENSOR_DTYPE_UINT32));
    return APP_ERR_OK;
}

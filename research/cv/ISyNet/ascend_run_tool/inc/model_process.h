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

#pragma once
#include <iostream>
#include <string>
#include "inc/utils.h"
#include "acl/acl.h"

using std::string;

class ModelProcess {
 public:
    /**
    * @brief Constructor
    */
    ModelProcess();

    /**
    * @brief Destructor
    */
    virtual ~ModelProcess();

    /**
    * @brief load model
    * @param [in] modelPath: model path
    * @return result
    */
    Result LoadModel(const char *modelPath);

    /**
    * @brief unload model
    */
    void UnloadModel();

    /**
    * @brief create model desc
    * @return result
    */
    Result CreateModelDesc();

    /**
    * @brief destroy desc
    */
    void DestroyModelDesc();

    /**
    * @get input size by index
    * @param [in] index: input index
    * @param [out] inputSize: input size of index
    * @return result
    */
    size_t GetInputSizeByIndex(const size_t index);

    /**
    * @brief create model input
    */
    Result CreateInput();

    /**
    * @brief add data to input
    * @param [in] inputDataBuffer: input buffer
    * @param [in] bufferSize: input buffer size
    * @return result
    */
    Result AddInput(void *data, size_t dataSize, int index);

    /**
    * @brief destroy input resource
    */
    void DestroyInput();

    /**
    * @brief create output buffer
    * @return result
    */
    Result CreateOutput();

    /**
    * @brief destroy output resource
    */
    void DestroyOutput();

    /**
    * @brief model execute
    * @return result
    */
    Result Execute();

    /**
    * @brief dump model output result to file
    */
    void DumpModelOutputResult(const string& base_dir);

    /**
    * @brief print model output result
    */
    void OutputModelResult();

    /**
    * @brief Get number of model's inputs
    */
    size_t getNumInputs() const;

    /**
    * @brief Get input name by the index
    */
    std::string getInputNameByIndex(size_t index) const;

 private:
    uint32_t modelId_;
    size_t modelWorkSize_;  // model work memory buffer size
    size_t modelWeightSize_;  // model weight memory buffer size
    void *modelWorkPtr_;  // model work memory buffer
    void *modelWeightPtr_;  // model weight memory buffer
    bool loadFlag_;  // model load flag
    aclmdlDesc *modelDesc_;
    aclmdlDataset *input_;
    aclmdlDataset *output_;
};


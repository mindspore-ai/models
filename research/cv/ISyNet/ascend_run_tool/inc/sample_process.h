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
#include <vector>

#include "inc/utils.h"
#include "acl/acl.h"

using std::string;
using std::vector;

class ModelProcess;

class SampleProcess {
 public:
    /**
    * @brief Constructor
    */
    SampleProcess();

    /**
    * @brief Destructor
    */
    virtual ~SampleProcess();

    /**
    * @brief init reousce
    * @return result
    */
    Result InitResource();

    /**
    * @brief Run inference loop of the model
    * @return result
    */
    Result RunInferenceLoop(ModelProcess* modelProcessPtr,
                            std::vector<void*>* all_buffers_ptr,
                            const std::vector<size_t>& buffer_sizes,
                            const std::vector<std::string>& all_dirs,
                            size_t num_inputs);

    /**
    * @brief sample process
    * @return result
    */
    Result Process();

    void freeAllBuffers(std::vector<void*>* all_buffers_ptr);

    void setResourcesBasePath(const std::string& path);

    void setOmPath(const std::string& path);

    void setOutputPath(const std::string& path);

    void setAclPath(const std::string& path);

 private:
    void DestroyResource();

    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;

    std::string resources_base_path_;
    std::string om_path_;
    std::string output_path_;
    std::string acl_path_;
};

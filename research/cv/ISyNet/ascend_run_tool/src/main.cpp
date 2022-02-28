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

#include <iostream>
#include "inc/sample_process.h"
#include "inc/utils.h"

bool g_isDevice = false;

int main(int argc, char** argv) {
    SampleProcess sampleProcess;
    if (argc < 5) {
        std::cout << std::string(argv[0])
                  << " /path/to/acl.json /base/path/to/input/data/ /path/to/om/file.om /base/path/to/output/data/"
                  << std::endl;
        return FAILED;
    }
    const string acl_json_path(argv[1]);
    const string resources_base_path(argv[2]);
    const string om_path(argv[3]);
    const string output_path(argv[4]);
    std::cout << "Acl JSON file: " << acl_json_path << std::endl;
    std::cout << "Resources base path: " << resources_base_path << std::endl;
    std::cout << "Om file: " << om_path << std::endl;
    std::cout << "Output path: " << output_path << std::endl;
    sampleProcess.setResourcesBasePath(resources_base_path);
    sampleProcess.setOmPath(om_path);
    sampleProcess.setOutputPath(output_path);
    sampleProcess.setAclPath(acl_json_path);
    Result ret = sampleProcess.InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("sample init resource failed");
        return FAILED;
    }

    ret = sampleProcess.Process();
    if (ret != SUCCESS) {
        ERROR_LOG("sample process failed");
        return FAILED;
    }

    INFO_LOG("execute sample success");
    return SUCCESS;
}

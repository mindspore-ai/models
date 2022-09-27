/*
* Copyright(C) 2022. Huawei Technologies Co.,Ltd
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

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include "c3dInfer.h"
#include <typeinfo>
#include "MxBase/Log/Log.h"


void create_dir(std::string path) {
    if (opendir(path.c_str()) == nullptr) {
        int isCreate = mkdir("./result", S_IRWXU);
        if (!isCreate) {
            std::cout << "Create dir success" << std::endl;
        } else {
            std::cout << "Create dir failed" << std::endl;
        }
    }
}

bool compare_pred(unsigned char a, unsigned char b) {
    return std::tolower(a) == std::tolower(b);
}
bool endsWith(const std::string& str, const std::string& suffix) {
    if (str.size() < suffix.size()) {
        return false;
    }
    std::string tstr = str.substr(str.size() - suffix.size());

    if (tstr.length() == suffix.length()) {
        return std::equal(suffix.begin(), suffix.end(), tstr.begin(), compare_pred);
    } else {
        return false;
    }
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input folder path, such as './mxBase ./test'.";
        return APP_ERR_OK;
    }
    LogInfo << "Project begin!!!!";
    LogInfo << argc;
    auto c3d = std::make_shared<c3dInfer>(0, "../data/model/c3d.om");
    std::string videoPath = argv[1];
    APP_ERROR ret = c3d->Process(videoPath);
    if (ret != APP_ERR_OK) {
        LogError << "c3dInfer process failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << std::endl;
    LogInfo << "Project end.";
    return APP_ERR_OK;
}

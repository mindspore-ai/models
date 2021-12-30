/*
* Copyright(C) 2021. Huawei Technologies Co.,Ltd
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
#include <ECOLiteInfer.h>
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NU = 1;
    const uint32_t BIASES_NU = 18;
    const uint32_t ANCHOR_DIM = 3;
    const uint32_t YOLO_TYPE = 3;
}

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
std::vector<std::string> get_files(const std::string &path) {
    DIR *dir = nullptr;
    std::vector<std::string> files;
    struct dirent *ptr = nullptr;

    if ((dir = opendir(path.c_str())) == nullptr) {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr = readdir(dir)) != nullptr) {
        if (ptr->d_type == 8) {
            files.push_back(path +"/"+ptr->d_name);
        }
    }
    closedir(dir);
    return files;
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
    create_dir("./result");
    std::vector<std::string> files;
    auto eco = std::make_shared<ECOLiteInfer>(0, "../convert/models/bs_16_ecolite.om");
    files = get_files(argv[1]);
    for (uint32_t i = 0; i < files.size(); i++) {
        // start of inference service
        std::string videoPath = files[i];
        std::string suffix = ".bin";
        if (endsWith(videoPath, suffix)) {
            LogInfo << "videoPath: " << videoPath;
            APP_ERROR ret = eco->Process(videoPath, "./result");
            if (ret != APP_ERR_OK) {
                LogError << "ECOLiteInfer process failed, ret=" << ret << ".";
                return ret;
            }
        }
    }
    LogInfo << "Project end";
    return APP_ERR_OK;
}

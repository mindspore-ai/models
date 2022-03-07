/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include <TALLInfer.h>
#include "MxBase/Log/Log.h"


void create_dir(std::string path) {
    DIR *dir = nullptr;
    if (opendir(path.c_str()) == nullptr) {
        int isCreate = mkdir("./result", S_IRWXU);
        if (!isCreate) {
            std::cout << "Create dir success" << std::endl;
        } else {
            std::cout << "Create dir failed" << std::endl;
        }
    }
}
void get_files(std::string path, std::vector<std::string> &files) {
    DIR *dir = nullptr;
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
    if (argc <= 2) {
        LogWarn << "Please input folder path, such as './mxBase ./test'.";
        return APP_ERR_OK;
    }
    LogInfo << argc;
    create_dir("./result");
    std::vector<std::string> files;
    auto tall = std::make_shared<TALLInfer>(0, argv[2]);

    get_files(argv[1], files);
    for (uint32_t i = 0; i < files.size(); i++) {
        std::string imgPath = files[i];
        std::string suffix = ".data";
        if (endsWith(imgPath, suffix)) {
            LogInfo << "videoPath: " << imgPath;
            APP_ERROR ret = tall->Process(imgPath, "./result");
            if (ret != APP_ERR_OK) {
                LogError << "TALLInfer process failed, ret=" << ret << ".";
                return ret;
            }
        }
    }
    LogInfo << "Project end";
    return APP_ERR_OK;
}

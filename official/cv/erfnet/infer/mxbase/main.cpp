/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd
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
#include <ERFNetInfer/erfnetinfer.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NU = 1;
    const uint32_t BIASES_NU = 18;
    const uint32_t ANCHOR_DIM = 3;
    const uint32_t YOLO_TYPE = 3;
}

void create_dir(std::string path) {
    DIR *dir = nullptr;
    dir = opendir(path.c_str());
    if (dir == nullptr) {
        int isCreate = mkdir("./result", S_IRWXU);
        if (!isCreate) {
            std::cout << "Create dir success" << std::endl;
        } else {
            std::cout << "Create dir failed" << std::endl;
        }
    }
}
void get_files(std::string path, std::vector<std::string> *files) {
    DIR *dir = nullptr;
    struct dirent *ptr = nullptr;

    if ((dir = opendir(path.c_str())) == nullptr) {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr = readdir(dir)) != nullptr) {
        if (ptr->d_type == 8) {
            files->push_back(path +"/"+ptr->d_name);
        }
    }
    closedir(dir);
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input folder path, such as './mxBase ./test'.";
        return APP_ERR_OK;
    }
    LogInfo << "Project begin!!!!";
    create_dir("./result");
    std::vector<std::string> files;
    auto erfnet = std::make_shared<ERFNetInfer>(0, "./model/ERFNet.om");

    get_files(argv[1], &files);
    for (uint32_t i = 0; i < files.size(); i++)  {
        // 推理业务开始
        std::string imgPath = files[i];
        APP_ERROR ret = erfnet->Process(imgPath, "./result");
        if (ret != APP_ERR_OK) {
            LogError << "ERFNetInfer process failed, ret=" << ret << ".";
            return ret;
        }
    }
    LogInfo << "Project end!?";
    return APP_ERR_OK;
}

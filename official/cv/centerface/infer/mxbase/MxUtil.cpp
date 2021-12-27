/*
 * Copyright (c) 2021.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "MxUtil.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <utility>
#include <vector>


void Soft_NMS(std::vector<MxBase::ObjectInfo>& vec_boxs, float sigma, float Nt,
              float threshold, unsigned int method) {
    int box_len = vec_boxs.size();
    for (int i = 0; i < box_len; i++) {
        MxBase::ObjectInfo* max_ptr = &vec_boxs[i];
        // get max box
        for (int pos = i + 1; pos < box_len; pos++)
            if (vec_boxs[pos].confidence > max_ptr->confidence)
                max_ptr = &vec_boxs[pos];

        // swap ith box with position of max box
        if (max_ptr != &vec_boxs[i]) std::swap(*max_ptr, vec_boxs[i]);

        max_ptr = &vec_boxs[i];

        for (int pos = i + 1; pos < box_len; pos++) {
            MxBase::ObjectInfo& curr_box = vec_boxs[pos];
            float area = (curr_box.x1 - curr_box.x0 + 1) *
                         (curr_box.y1 - curr_box.y0 + 1);
            float iw = std::min(max_ptr->x1, curr_box.x1) -
                       std::max(max_ptr->x0, curr_box.x0) + 1;
            float ih = std::min(max_ptr->y1, curr_box.y1) -
                       std::max(max_ptr->y0, curr_box.y0) + 1;
            if (iw > 0 && ih > 0) {
                float overlaps = iw * ih;
                // iou between max box and detection box
                float iou = overlaps / ((max_ptr->x1 - max_ptr->x0 + 1) *
                                            (max_ptr->y1 - max_ptr->y0 + 1) +
                                        area - overlaps);
                float weight = 0;
                if (method == 1)  // linear
                    weight = iou > Nt ? 1 - iou : 1;
                else if (method == 2)  // gaussian
                    weight = std::exp(-(iou * iou) / sigma);
                else  // original NMS
                    weight = iou > Nt ? 0 : 1;
                // adjust all bbox score after this box
                curr_box.confidence *= weight;
                // if new confidence less then threshold , swap with last one
                // and shrink this array
                if (curr_box.confidence < threshold) {
                    std::swap(curr_box, vec_boxs[box_len - 1]);
                    box_len--;
                    pos--;
                }
            }
        }
    }
}

std::string ResolvePathName(const std::string& filepath) {
    size_t npos = filepath.rfind('/');
    if (npos == std::string::npos) return std::string();
    return filepath.substr(0, npos);
}

bool FetchDirFiles(const std::string& filepath,
                   std::vector<std::string>& files) {
    DIR* dir = opendir(filepath.c_str());
    if (!dir) {
        LOG_SYS_ERROR("opendir for path:" << filepath);
        return false;
    }
    dirent* dirFile = readdir(dir);
    while (dirFile) {
        if (strcmp(dirFile->d_name, ".") == 0 ||
            strcmp(dirFile->d_name, "..") == 0) {
            dirFile = readdir(dir);
            continue;
        }
        std::string path = filepath + "/" + dirFile->d_name;
        if (D_ISREG(dirFile->d_type)) {
            files.emplace_back(path);
        } else if (D_ISDIR(dirFile->d_type)) {
            if (!FetchDirFiles(path, files)) return false;
        }
        dirFile = readdir(dir);
    }
    closedir(dir);
    return true;
}

bool MkdirRecursive(const std::string& filepath) {
    int ret = mkdir(filepath.c_str(), 0777);
    if (ret == 0 || errno == EEXIST) {
        return true;
    } else {
        std::string parent = ResolvePathName(filepath);
        if (parent.empty() || !MkdirRecursive(parent)) return false;
        return mkdir(filepath.c_str(), 0777) == 0;
    }
}

bool FetchTestFiles(std::string& imagePath, std::vector<std::string>& files) {
    char absPath[PATH_MAX];
    if (realpath(imagePath.c_str(), absPath) == nullptr) {
        LOG_SYS_ERROR("get the file real path:" << imagePath);
        return false;
    }
    imagePath = absPath;
    struct stat buffer;
    if (stat(absPath, &buffer) != 0) {
        LOG_SYS_ERROR("stat file path:" << absPath);
        return false;
    }
    if (S_ISREG(buffer.st_mode)) {
        files.push_back(absPath);
        imagePath = ResolvePathName(imagePath);
        return true;
    } else if (S_ISDIR(buffer.st_mode)) {
        return FetchDirFiles(imagePath, files);
    } else {
        LogFatal << "not a regular file or dir path !!!";
        return false;
    }
}

APP_ERROR ReadFileToMem(const std::string& filePath, std::string& mem) {
    std::ifstream file(filePath.c_str(), std::ifstream::binary);
    if (!file) {
        LogError << "open file for read error:" << filePath;
        return -1;
    }
    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);

    mem.resize(fileSize);
    file.read(&mem[0], fileSize);
    file.close();
    return APP_ERR_OK;
}

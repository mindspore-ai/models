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

#ifndef OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXUTIL_H_
#define OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXUTIL_H_

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <string>
#include <vector>

#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "MxBase/Log/Log.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"

#define LOG_SYS_ERROR(msg) LogError << msg << " error:" << strerror(errno)
#define D_ISREG(ty) ((ty) == DT_REG)
#define D_ISDIR(ty) ((ty) == DT_DIR)

void Soft_NMS(std::vector<MxBase::ObjectInfo>& vec_boxs, float sigma = 0.5,
              float Nt = 0.5, float threshold = 0.001, unsigned int method = 2);

std::string ResolvePathName(const std::string& filepath);

bool FetchDirFiles(const std::string& filepath,
                   std::vector<std::string>& files);

bool MkdirRecursive(const std::string& filepath);

bool FetchTestFiles(std::string& imagePath, std::vector<std::string>& files);

APP_ERROR ReadFileToMem(const std::string& filePath, std::string& mem);

#endif  // OFFICIAL_CV_CENTERFACE_INFER_MXBASE_MXUTIL_H_

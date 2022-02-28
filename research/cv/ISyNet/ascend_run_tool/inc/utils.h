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
#include <string>
#include <vector>
#include <iostream>

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stdout, "[ERROR] " fmt "\n", ##__VA_ARGS__)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

class Utils {
 public:
    /**
    * @brief create buffer of file
    * @param [in] fileName: file name
    * @param [out] inputBuff: input data buffer
    * @param [out] fileSize: size of file
    * @return result
    */
    static Result ReadBinFile(const std::string &fileName, void **inputBuffPtr, uint32_t *fileSizePtr);

    /**
    * @brief create buffer of file
    * @param [in] fileName: file name
    * @param [out] picDevBuffer: input data device buffer which need to be memcpy
    * @param [out] inputBuffSize: size of inputBuff
    * @return result
    */
    static Result MemcpyFileToDeviceBuffer(const std::string &fileName, void *picDevBuffer, size_t inputBuffSize);

    /**
    * @brief Check whether the path is a file.
    * @param [in] fileName: fold to check
    * @return result
    */
    static Result CheckPathIsFile(const std::string &fileName);

    static std::vector<std::string> ListDir(const std::string& dir);

    static bool folder_exists(const std::string& dir);

    static void mkdir(const std::string& dir);
};

#pragma once

/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef OFFICIAL_RECOMMEND_WIDE_AND_DEEP_INFER_SDK_SAMPLE_COMMANDFLAGPARSER_H_
#define OFFICIAL_RECOMMEND_WIDE_AND_DEEP_INFER_SDK_SAMPLE_COMMANDFLAGPARSER_H_

#include <iostream>
#include <map>
#include <string>

#include "MxBase/Log/Log.h"

#define DEFINE_string(name, default_arg, desc)                             \
    static std::string FLAGS_##name(default_arg);                          \
    static OptionManager::OptionInitializer g_string_##name##_initializer( \
        &FLAGS_##name, #name, "string", #desc);

#define DEFINE_int32(name, default_arg, desc)                               \
    static int32_t FLAGS_##name(default_arg);                               \
    static OptionManager::OptionInitializer g_int32_t_##name##_initializer( \
        &FLAGS_##name, #name, "int32_t", #desc);

class OptionManager {
 private:
    struct OptInfo {
        std::string name;
        std::string type;
        std::string desc;
        void* ptr;

        // a helper function to convert a command line argument to its correct
        // type
        void SetValue(const std::string &type, const std::string &val) {
            if (type == "string") {
                *(std::string*)ptr = val;
            } else if (type == "int32_t") {
                *reinterpret_cast<int32_t*>(ptr) = stoi(val);
            }
        }
    };

    std::map<std::string, OptInfo> flag_table;

    // to use the singleton pattern, define the constructor as private
    OptionManager() {}

 public:
    class OptionInitializer {
     public:
        OptionInitializer(void* var_ptr, std::string name, std::string type,
                          std::string desc) {
            OptionManager::getInstance()->registerOption(var_ptr, name, type,
                                                         desc);
        }
    };

    static OptionManager* getInstance() {
        static OptionManager g_om_instance;
        return &g_om_instance;
    }

    void registerOption(void* var_ptr, std::string name, std::string type,
                        std::string desc) {
        flag_table[name] = {name, type, desc, var_ptr};
    }

    void parseCommandLineFlags(int argc, char** argv) {
        std::string arg_name;
        std::string arg_val;
        for (int index = 1; index < argc; index++) {
            if (argv[index][0] == '-' && argv[index][1] == '-') {
                arg_name = argv[index] + 2;
                if (index + 1 >= argc ||
                    (argv[index + 1][0] == '-' && argv[index + 1][1] == '-')) {
                    arg_val.clear();
                } else {
                    arg_val = argv[index + 1];
                    index++;
                }
            } else {
                std::cout << "invalid argument list !!!" << std::endl;
                exit(-1);
            }
            if (flag_table.count(arg_name) != 0) {
                if (flag_table[arg_name].type == "string") {
                    *(std::string*)flag_table[arg_name].ptr = arg_val;
                } else if (flag_table[arg_name].type == "int32_t") {
                    *reinterpret_cast<int32_t*>(flag_table[arg_name].ptr) =
                        stoi(arg_val);
                }

            } else {
                std::cout << "unrecognized command flag: " << arg_name
                          << std::endl;
            }
        }
    }
};

#endif  // OFFICIAL_RECOMMEND_WIDE_AND_DEEP_INFER_SDK_SAMPLE_COMMANDFLAGPARSER_H_

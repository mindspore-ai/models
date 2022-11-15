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

#ifndef MINDSPORE_MODELS_UTILS_FLAG_PARSER_H_
#define MINDSPORE_MODELS_UTILS_FLAG_PARSER_H_

#include <map>
#include <vector>
#include <utility>
#include <string>
#include <sstream>

// declare
#define DEFINE_string(name, default_val, desc)   \
  static std::string FLAGS_##name = default_val; \
  static _FlagsReg reg_flag_##name(#name, desc, &FLAGS_##name);

#define DEFINE_int32(name, default_val, desc) \
  static int32_t FLAGS_##name = default_val;  \
  static _FlagsReg reg_flag_##name(#name, desc, &FLAGS_##name);

#define DEFINE_bool(name, default_val, desc) \
  static bool FLAGS_##name = default_val;    \
  static _FlagsReg reg_flag_##name(#name, desc, &FLAGS_##name);

bool ParseCommandLineFlags(int argc, char **argv);
bool ParseCommandLineFlags(int *argc, char ***argv);

// implement
enum _FlagType {
  _FlagTypeString = 0,
  _FlagTypeInt32 = 1,
  _FlagTypeBool = 16,
};

struct _FlagInfo {
  std::string name;
  std::string desc;
  _FlagType type = _FlagTypeString;

  std::string *str_val = nullptr;
  int32_t *int32_val = nullptr;
  bool *bool_val = nullptr;
};

class _FlagsStorage {
 public:
  static _FlagsStorage &Instance() {
    static _FlagsStorage instance;
    return instance;
  }
  bool Reg(const _FlagInfo &info) {
    if (GetFlag(info.name) != nullptr) {
      std::cout << "Flag " << info.name << " is repeat registered" << std::endl;
      return false;
    }
    flags.emplace_back(info);
    return true;
  }
  _FlagInfo *GetFlag(const std::string &name) {
    for (auto &item : flags) {
      if (item.name == name) {
        return &item;
      }
    }
    return nullptr;
  }
  std::vector<_FlagInfo> flags;
};

class _FlagsReg {
 public:
  _FlagsReg(const std::string &name, const std::string &desc, std::string *var) {
    _FlagInfo info;
    info.name = name;
    info.desc = desc;
    info.type = _FlagTypeString;
    info.str_val = var;
    _FlagsStorage::Instance().Reg(info);
  }
  _FlagsReg(const std::string &name, const std::string &desc, int32_t *var) {
    _FlagInfo info;
    info.name = name;
    info.desc = desc;
    info.type = _FlagTypeInt32;
    info.int32_val = var;
    _FlagsStorage::Instance().Reg(info);
  }
  _FlagsReg(const std::string &name, const std::string &desc, bool *var) {
    _FlagInfo info;
    info.name = name;
    info.desc = desc;
    info.type = _FlagTypeBool;
    info.bool_val = var;
    _FlagsStorage::Instance().Reg(info);
  }
};

inline void Trim(std::string *input) {
  if (input == nullptr) {
    return;
  }
  if (input->empty()) {
    return;
  }

  const char WHITESPACE[] = "\t\n\v\f\r ";
  input->erase(0, input->find_first_not_of(WHITESPACE));
  input->erase(input->find_last_not_of(WHITESPACE) + 1);
}

// get the file name from a given path
// for example: "/usr/bin", we will get "bin"
inline std::string GetFileName(const std::string &path) {
  if (path.empty()) {
    return "";
  }
  char delim = '/';
  size_t i = path.rfind(delim, path.length());
  if (i != std::string::npos && i + 1 < path.length()) {
    return (path.substr(i + 1, path.length() - i));
  }
  return "";
}

template <typename T>
inline bool GenericParseValue(const std::string &value, T *ret_val) {
  T ret;
  std::istringstream input(value);
  input >> ret;

  if (input && input.eof()) {
    *ret_val = ret;
    return true;
  }
  return false;
}

bool ParseCommandLineFlags(int argc, char **argv) {
  if (argv == nullptr) {
    return false;
  }
  const int FLAG_PREFIX_LEN = 2;
  if (argc <= 0) {
    std::cout << "The arguments number is out of range";
    return false;
  }
  std::string binName = GetFileName(argv[0]);

  for (int i = 1; i < argc; i++) {
    std::string tmp = argv[i];
    Trim(&tmp);
    const std::string flagItem(tmp);

    if (flagItem == "--") {
      break;
    }

    if (flagItem.find("--") == std::string::npos) {
      std::cout << "Failed: flag " + flagItem + " is not valid." << std::endl;
      return false;
    }
    size_t pos = flagItem.find_first_of('=');
    if (pos == std::string::npos) {
      std::cout << "Failed: flag " + flagItem + " is not valid." << std::endl;
      return false;
    }
    std::string key = flagItem.substr(FLAG_PREFIX_LEN, pos - FLAG_PREFIX_LEN);
    std::string value = flagItem.substr(pos + 1);
    auto info = _FlagsStorage::Instance().GetFlag(key);
    if (info == nullptr) {
      std::cout << "Cannot find flag " << key << " registered." << std::endl;
      continue;
    }
    switch (info->type) {
      case _FlagTypeString:
        if (info->str_val == nullptr) {
          std::cout << "Inner error: invalid str flag " << key << std::endl;
          return false;
        }
        *info->str_val = value;
        break;
      case _FlagTypeInt32:
        if (info->int32_val == nullptr) {
          std::cout << "Inner error: invalid int32 flag " << key << std::endl;
          return false;
        }
        if (!GenericParseValue(value, info->int32_val)) {
          std::cout << "Invalid int32 flag " << key << " value: " << value << std::endl;
          return false;
        }
        break;
      case _FlagTypeBool:
        if (info->bool_val == nullptr) {
          std::cout << "Inner error: invalid bool flag " << key << std::endl;
          return false;
        }
        if (value == "true") {
          *info->bool_val = true;
        } else if (value == "false") {
          *info->bool_val = false;
        } else {
          std::cout << "Invalid bool flag " << key << " value: " << value << std::endl;
          return false;
        }
        break;
    }
  }
  return true;
}

bool ParseCommandLineFlags(int *argc, char ***argv) {
  if (argc == nullptr || argv == nullptr) {
    return false;
  }
  return ParseCommandLineFlags(*argc, *argv);
}
#endif  // MINDSPORE_MODELS_UTILS_FLAG_PARSER_H_

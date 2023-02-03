#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
cmake_minimum_required(VERSION 3.14.1)
project(CppInfer)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -std=c++17 -Werror -Wall -fPIE -Wl,--allow-shlib-undefined")
set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR}/)
option(MINDSPORE_PATH "mindspore install path" "")
include_directories(${MINDSPORE_PATH})
include_directories(${MINDSPORE_PATH}/include)

set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
include_directories(${TOP_DIR}/utils/cpp_infer/example/)  # common_inc in top dir
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)  # common_inc in local dir
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)  # common_inc in local dir
include_directories(../third_party/src)

if(EXISTS ${MINDSPORE_PATH}/lib/libmindspore-lite.so)
    message(--------------- Compile-with-MindSpore-Lite ----------------)
    set(MS_LIB ${MINDSPORE_PATH}/lib/libmindspore-lite.so)
    set(MD_LIB ${MINDSPORE_PATH}/lib/libminddata-lite.so)
    add_compile_definitions(ENABLE_LITE)
else()
    message(--------------- Compile-with-MindSpore ----------------)
    set(MS_LIB ${MINDSPORE_PATH}/lib/libmindspore.so)
    file(GLOB_RECURSE MD_LIB ${MINDSPORE_PATH}/_c_dataengine*)
endif()

add_executable(main src/main.cc)
target_link_libraries(main ${MS_LIB} ${MD_LIB})

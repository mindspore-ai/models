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

set -e

# curr path
cur_path=$(
  cd "$(dirname "$0")" || exit
  pwd
)
# env ready
env_ready=true

# execute arguments
input_sentences_dir=''
max_load_num=''
evaluate=false
label_path=''

function parse_arguments() {
  echo "parsing arguments ..."
  while getopts "i:m:e:l:" opt; do
    case ${opt} in
    i)
      input_sentences_dir=$OPTARG
      ;;
    m)
      max_load_num=$OPTARG
      ;;
    e)
      evaluate=$OPTARG
      ;;
    l)
      label_path=$OPTARG
      ;;
    *)
      echo "*分支:${OPTARG}"
      ;;
    esac
  done

  # print arguments
  echo "---------------------------"
  echo "| execute arguments"
  echo "| input sentences dir: $input_sentences_dir"
  echo "| max load num: $max_load_num"
  echo "| evaluate: $evaluate"
  echo "| input sentences label path: $label_path"
  echo "---------------------------"
}

function check_env() {
  echo "checking env ..."
  # check MindXSDK env
  if [ ! "${MX_SDK_HOME}" ]; then
    env_ready=false
    echo "please set MX_SDK_HOME path into env."
  else
    echo "MX_SDK_HOME set as ${MX_SDK_HOME}, ready."
  fi
}

function execute() {
  if [ "${max_load_num}" == '' ]; then
    if [ "${evaluate}" == true ]; then
        ./lstm "$input_sentences_dir" 1 "$label_path"
    else
        ./lstm "$input_sentences_dir"
    fi
  else
    if [ "${evaluate}" == true ]; then
        ./lstm "$input_sentences_dir" "$max_load_num" 1 "$label_path"
    else
        ./lstm "$input_sentences_dir" "$max_load_num"
    fi
  fi
}

function run() {
  echo -e "\ncurrent dir: $cur_path"
  # parse arguments
  parse_arguments "$@"

  # check environment
  check_env
  if [ "${env_ready}" == false ]; then
    echo "please set env first."
    exit 0
  fi

  # check file
  if [ ! -f "${cur_path}/lstm" ]; then
    echo "lstm not exist, please build first."
    exit 0
  fi

  echo "---------------------------"
  echo "prepare to execute program."
  echo -e "---------------------------\n"

  execute
}

run "$@"
exit 0

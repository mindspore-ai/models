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
# param check
infer_param_ready=true
eval_param_ready=true

# execute arguments

# execute mode opt = {'infer', 'evaluate'}
mode='infer'
# input sentence dir, effective in infer mode
input=''
# vocab path, effective in infer mode
vocab=''
# use preprocess sentences as input, opt = {true, false}
use_preprocess=false
# number of loading infer files
load_preprocess_num=''
# test set feature path, effective in evaluate mode
feature=''
# test set label path, effective in evaluate mode
label=''
# use default config
use_default=false

function parse_arguments() {
  echo "parsing arguments ..."
  while getopts "m:i:v:p:n:f:l:u:" opt; do
    case ${opt} in
    m)
      mode=$OPTARG
      ;;
    i)
      input=$OPTARG
      ;;
    v)
      vocab=$OPTARG
      ;;
    p)
      use_preprocess=$OPTARG
      ;;
    n)
      load_preprocess_num=$OPTARG
      ;;
    f)
      feature=$OPTARG
      ;;
    l)
      label=$OPTARG
      ;;
    u)
      use_default=$OPTARG
      ;;
    *)
      echo "*分支:${OPTARG}"
      ;;
    esac
  done

  # print arguments
  echo "---------------------------"
  echo "| execute arguments"
  echo "| mode: $mode"
  echo "| input sentence dir: $input"
  echo "| vocab path: $vocab"
  echo "| use preprocess sentences: $use_preprocess"
  echo "| load preprocess num: $load_preprocess_num"
  echo "| test set feature path: $feature"
  echo "| test set label path: $label"
  echo "| use default config: $use_default"
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

function check_infer_param() {
    if [ "$input" == '' ]; then
        infer_param_ready=false
        echo "please config input sentence dir"
    fi
    if [ "$use_preprocess" == false ]; then
      if [ "$vocab" == '' ]; then
          infer_param_ready=false
          echo "please config vocab path"
      fi
    else
      if [ "$load_preprocess_num" == '' ]; then
          infer_param_ready=false
          echo "please config load preprocess num"
      fi
    fi
}

function check_eval_param() {
  if [ "$feature" == '' ]; then
    eval_param_ready=false
    echo "please config test set feature path"
  fi
  if [ "$label" == '' ]; then
    eval_param_ready=false
    echo "please config test set label path"
  fi
}

function execute() {
  if [ "${mode}" == 'infer' ]; then
    if [ "$use_default" == true ]; then
        python3.7 main.py --parse_word_vector=true
    else
      check_infer_param
      if [ "$infer_param_ready" == false ]; then
          echo "please check infer parameters"
          exit 0
      fi
      if [ "$use_preprocess" == true ]; then
        python3.7 main.py --input_sentences_dir="$input" --max_load_num="$load_preprocess_num"
      else
        python3.7 main.py --input_sentences_dir="$input" --vocab_path="$vocab" --parse_word_vector=true
      fi
    fi
  elif [ "${mode}" == 'evaluate' ]; then
     if [ "$use_default" == true ]; then
        python3.7 main.py --do_eval=true
    else
      check_eval_param
      if [ "$eval_param_ready" == false ]; then
        echo "please check evaluate parameters"
        exit 0
      fi
      python3.7 main.py --do_eval=true --test_set_feature_path="$feature" --test_set_label_path="$label" --max_load_num="$load_preprocess_num"
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

  echo "---------------------------"
  echo "prepare to execute program."
  echo -e "---------------------------\n"

  execute
}

run "$@"
exit 0

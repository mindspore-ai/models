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

# an simple tutorial as follows, more parameters can be setting
if [ $# != 3 ]
then
    echo "Usage: bash run_eval_ascend.sh [permuted_mnist|adding_problem] [DATA_PATH] [CKPT_FILE]"
exit 1
fi

export DATASET_NAME=$1
export DATA_PATH=$2
export CKPT_FILE=$3

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

if [ $# -ge 1 ]; then
  if [ $1 == 'adding_problem' ]; then
    CONFIG_FILE="../../config_addingproblem.yaml"
  elif [ $1 == 'permuted_mnist' ]; then
    CONFIG_FILE="../../default_config.yaml"
  else
    echo "Unrecognized parameter"
    exit 1
  fi
else
  CONFIG_FILE="../../default_config.yaml"
fi

python -s ${BASE_PATH}/../eval.py --config_path=$CONFIG_FILE --test_data_path=$DATA_PATH --device_target="Ascend" --ckpt_file=$CKPT_FILE > eval.log 2>&1 &
               

#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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
usage() {
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash select_sample.sh DEVICE_NUM EXP_DIR ANNOTATION PRE_TRAINED(option)"
  echo "for example: bash select_sample.sh 8 /path/to/save/folder /path/to/annotation.json /path/to/model_after_step1.ckpt "
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
}

if [ $# -lt 3 ]; then
  usage
  exit 1
fi

ulimit -c unlimited
ulimit -s unlimited
ulimit -u unlimited

DEVICE_NUM=$1
EXP_DIR=$2
ANNOTATION=$3
PRE_TRAINED=$4
echo "start training on $DEVICE_NUM devices"

# GPU
mpirun --allow-run-as-root -n $DEVICE_NUM \
    python3 ../select_sample.py  \
    --is_distributed=True \
    --device_target="GPU"\
    --exp_dir=$EXP_DIR \
    --pre_trained=$PRE_TRAINED \
    --annotation=$ANNOTATION


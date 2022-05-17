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
  echo "bash run_distribute_train_model_gpu.sh DEVICE_NUM EXP_DIR ANNOTATION(option) PRE_TRAINED(option)"
  echo "for example: bash run_distribute_train_model_gpu.sh 8 /path/to/save/checkpoint/folder /path/to/annotation.json /path/to/pre_trained.ckpt"
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
}


if [ $# -lt 2 ]; then
  usage
  exit 1
elif [ $# -eq 3 ] && [[ $4 = *.ckpt ]]; then
    PRE_TRAINED=$3
    ANNOTATION=$4
else
    ANNOTATION=$3
    PRE_TRAINED=$4
fi

DEVICE_NUM=$1
EXP_DIR=$2

# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=3

mpirun --allow-run-as-root -n $DEVICE_NUM --output-filename log_output_GPU \
    python3 ../train.py  \
    --is_distributed=True \
    --device_target="GPU"\
    --exp_dir=$EXP_DIR \
    --annotation=$ANNOTATION \
    --pre_trained=$PRE_TRAINED
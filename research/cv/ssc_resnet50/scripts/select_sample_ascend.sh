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
  echo "bash select_sample.sh DEVICE_NUM EXP_DIR RANK_TABLE_FILE ANNOTATION PRE_TRAINED"
  echo "for example: bash select_sample_ascend.sh 8 /path/to/save/folder /path/to/rank_table_file.json /path/to/annotation.json /path/to/model_after_step1.ckpt"
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
}

if [ $# -lt 4 ]; then
  usage
  exit 1
fi

ulimit -c unlimited
ulimit -s unlimited
ulimit -u unlimited

DEVICE_NUM=$1
EXP_DIR=$2
RANK_TABLE_FILE=$3
ANNOTATION=$4
PRE_TRAINED=$5
echo "start training on $DEVICE_NUM devices"

export HCCL_CONNECT_TIMEOUT=7200 # 7200 is the max
export DEVICE_NUM=$DEVICE_NUM
export RANK_SIZE=$DEVICE_NUM
export RANK_TABLE_FILE=$RANK_TABLE_FILE


for((i=1; i<${DEVICE_NUM}; i++))
do
  export DEVICE_ID=$i
  export RANK_ID=$i
  # Ascend
  python ../select_sample.py \
         --is_distributed=True \
         --device_target="Ascend"\
         --exp_dir=$EXP_DIR \
         --pre_trained=$PRE_TRAINED \
         --annotation=$ANNOTATION &
done

export DEVICE_ID=0
export RANK_ID=0
# Ascend
python ../select_sample.py \
       --is_distributed=True \
       --device_target="Ascend"\
       --exp_dir=$EXP_DIR \
       --pre_trained=$PRE_TRAINED \
       --annotation=$ANNOTATION

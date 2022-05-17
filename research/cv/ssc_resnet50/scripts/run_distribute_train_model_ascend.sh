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
  echo "bash run_distribute_train_model_ascend.sh DEVICE_NUM EXP_DIR RANK_TABLE_FILE ANNOTATION(option) PRE_TRAINED(option)"
  echo "for example: bash run_distribute_train_model_ascend.sh 8 /path/to/save/checkpoint/folder /path/to/rank_table_file.json /path/to/annotation.json /path/to/pre_trained.ckpt"
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
}


if [ $# -lt 3 ]; then
  usage
  exit 1
elif [ $# -eq 4 ] && [[ $4 = *.ckpt ]]; then
    PRE_TRAINED=$4
    ANNOTATION=$5
else
    ANNOTATION=$4
    PRE_TRAINED=$5
fi

DEVICE_NUM=$1
EXP_DIR=$2
RANK_TABLE_FILE=$3

# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=3
export PYTHONUNBUFFERED=1

ulimit -u unlimited
export HCCL_CONNECT_TIMEOUT=7200 # 7200 is the max
export DEVICE_NUM=$DEVICE_NUM
export RANK_SIZE=$DEVICE_NUM
export RANK_TABLE_FILE=$RANK_TABLE_FILE

export SERVER_ID=0

current_time=$(date +%Y%m%d-%H%M%S)
prefix_dir="/train_${DEVICE_NUM}_${current_time}"

for((i=1; i<${DEVICE_NUM}; i++))
do
  export DEVICE_ID=$i
  export RANK_ID=$i

  mkdir -p ./${prefix_dir}/device$i

  cp -r ../src/ ../*.py ./${prefix_dir}/device$i
  cp ../*.json ./${prefix_dir}/device$i
  cp ../*.yaml ./${prefix_dir}/device$i

  cd ./${prefix_dir}/device$i || exit

  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  env > env.log

  python3 train.py \
          --device_target="Ascend" \
          --is_distributed=True \
          --exp_dir=$EXP_DIR \
          --annotation=$ANNOTATION \
          --pre_trained=$PRE_TRAINED > log.log 2>&1 &

  cd ../..
done

export DEVICE_ID=0
export RANK_ID=0

mkdir -p ./${prefix_dir}/device0

cp -r ../src/ ../*.py ./${prefix_dir}/device0
cp ../*.json ./${prefix_dir}/device0
cp ../*.yaml ./${prefix_dir}/device0

cd ./${prefix_dir}/device0 || exit

echo "start training for rank $RANK_ID, device $DEVICE_ID"
env > env.log

python3 train.py \
        --device_target="Ascend" \
        --is_distributed=True \
        --exp_dir=$EXP_DIR \
        --annotation=$ANNOTATION \
        --pre_trained=$PRE_TRAINED

cd ../..

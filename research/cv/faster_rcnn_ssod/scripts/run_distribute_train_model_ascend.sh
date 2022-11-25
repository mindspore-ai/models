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
  echo "bash run_distribute_train_model_ascend.sh [RANK_TABLE_FILE] [DATA_ROOT] [TRAIN_ANN_FILE] [OUTPUT_DIR] [PRE_TRAINED](option)"
  echo "For example: bash run_distribute_train_model_ascend.sh /home/rank_table_file.json /home/coco/images/ /home/coco/annotations/train_15.json /home/output /home/faster_rcnn-12_7393.ckpt"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
}

# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=3
export PYTHONUNBUFFERED=1

if [ $# -lt 4 ]; then
  usage
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_ROOT=$2
ANN_FILE=$3
OUTPUT_DIR=$4
PRE_TRAINED=$5

rank_table_file=$(get_real_path $1)
if [ ! -f $rank_table_file ]; then
  echo -e "error: RANK_TABLE_FILE=$rank_table_file is not a file \n"
  exit 1
fi

export RANK_TABLE_FILE=$rank_table_file
export RANK_SIZE=8
export DEVICE_START=0
export HCCL_CONNECT_TIMEOUT=6000

current_time=$(date +%Y%m%d-%H%M%S)
prefix_dir="$(pwd)/train_${RANK_SIZE}p_${current_time}"

ulimit -u unlimited
for ((i = ${DEVICE_START} + 1; i < (${DEVICE_START} + ${RANK_SIZE}); i++)); do
  echo "Process: device$i ..."
  mkdir -p ${prefix_dir}/device$i
  mkdir -p ${prefix_dir}/device$i/outputs/
  cp ../train.py ${prefix_dir}/device$i
  cp -r ../src/ ${prefix_dir}/device$i
  cd ${prefix_dir}/device$i
  export DEVICE_ID=$i
  export RANK_ID=$(($i - ${DEVICE_START}))
  echo "start training for device $i"
  env >env_log.txt
  python ./train.py --device_target=Ascend \
                            --device_id=${DEVICE_ID} \
                            --train_img_dir=$DATA_ROOT \
                            --train_ann_file=$ANN_FILE \
                            --save_checkpoint_path=$OUTPUT_DIR \
                            --pre_trained=$PRE_TRAINED \
                            --run_distribute >log.txt 2>&1 &
  cd -
done

echo "Process: device0 ..."
mkdir -p ${prefix_dir}/device0
mkdir -p ${prefix_dir}/device0/outputs/
cp ../train.py ${prefix_dir}/device0
cp -r ../src/ ${prefix_dir}/device0
cd ${prefix_dir}/device0
export DEVICE_ID=${DEVICE_START}
export RANK_ID=0
echo "start training for device 0"
env >env_log.txt
python ./train.py --device_target=Ascend \
                          --device_id=${DEVICE_ID} \
                          --train_img_dir=$DATA_ROOT \
                          --train_ann_file=$ANN_FILE \
                          --save_checkpoint_path=$OUTPUT_DIR \
                          --pre_trained=$PRE_TRAINED \
                          --run_distribute

echo -e "Process start OK, save in ${prefix_dir} \n"

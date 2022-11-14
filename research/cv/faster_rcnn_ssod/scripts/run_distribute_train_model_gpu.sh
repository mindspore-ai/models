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
  echo "bash run_distribute_train_gpu.sh [DATA_ROOT] [ANN_FILE] [OUTPUT_DIR] [PRE_TRAINED](option)"
  echo "For example: bash run_distribute_train_gpu.sh /home/coco/images/ /home/coco/annotations/train_25.json /home/outputs /home/faster_rcnn-12_7393.ckpt"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
}

# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=3
export PYTHONUNBUFFERED=1

if [ $# -lt 3 ]; then
  usage
  exit 1
fi

DATA_ROOT=$1
ANN_FILE=$2
OUTPUT_DIR=$3
PRE_TRAINED=$4

export RANK_SIZE=8
export DEVICE_ID=0

current_time=$(date +%Y%m%d-%H%M%S)
prefix_dir="$(pwd)/train_${RANK_SIZE}p_${current_time}"

ulimit -u unlimited

echo "Process: device${RANK_SIZE}p ..."
mkdir -p ${prefix_dir}/device${RANK_SIZE}p
mkdir -p ${prefix_dir}/device${RANK_SIZE}p/outputs/train_weights/
cp ../train.py ${prefix_dir}/device${RANK_SIZE}p
cp -r ../src/ ${prefix_dir}/device${RANK_SIZE}p
cd ${prefix_dir}/device${RANK_SIZE}p
echo "start training for device${RANK_SIZE}p"
env >env_log.txt
mpirun --allow-run-as-root -n $RANK_SIZE python ./train.py \
                                                --device_target=GPU \
                                                --train_img_dir=$DATA_ROOT \
                                                --train_ann_file=$ANN_FILE \
                                                --save_checkpoint_path=$OUTPUT_DIR \
                                                --pre_trained=$PRE_TRAINED \
                                                --run_distribute
cd -

echo -e "Process start OK, save in ${prefix_dir} \n"

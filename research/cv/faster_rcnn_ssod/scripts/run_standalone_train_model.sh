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
  echo "bash run_distribute_train_combine.sh [DEVICE_TARGET] [DEVICE_ID] [DATA_ROOT] [ANN_FILE] [OUTPUT_DIR] [PRE_TRAINED](option)"
  echo "For example: bash run_standalone_train_model.sh Ascend 0 /home/coco/images/ /home/coco/annotations/train_15.json /home/outputs /home/faster_rcnn-12_7393.ckpt"
  echo "It is better to use the absolute path."
  echo "=============================================================================================================="
}

# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=3
export PYTHONUNBUFFERED=1

if [ $# -lt 5 ]; then
  usage
  exit 1
fi


ulimit -u unlimited

DEVICE_TARGET=$1
DATA_ROOT=$3
ANN_FILE=$4
OUTPUT_DIR=$5
PRE_TRAINED=$6

export DEVICE_ID=$2

current_time=$(date +%Y%m%d-%H%M%S)
prefix_dir="$(pwd)/train_1p_${current_time}"

ulimit -u unlimited
echo "Process: device${DEVICE_ID} ..."
mkdir -p ${prefix_dir}/device${DEVICE_ID}
mkdir -p ${prefix_dir}/device${DEVICE_ID}/outputs/
cp ../train.py ${prefix_dir}/device${DEVICE_ID}
cp -r ../src/ ${prefix_dir}/device${DEVICE_ID}
cd ${prefix_dir}/device${DEVICE_ID}
echo "start training for device ${DEVICE_ID}"
env >env_log.txt
python ./train.py --device_id=${DEVICE_ID} \
                  --device_target=$DEVICE_TARGET \
                  --train_img_dir=$DATA_ROOT \
                  --train_ann_file=$ANN_FILE \
                  --save_checkpoint_path=$OUTPUT_DIR \
                  --pre_trained=$PRE_TRAINED
cd -

echo -e "Process start OK, save in ${prefix_dir} \n"

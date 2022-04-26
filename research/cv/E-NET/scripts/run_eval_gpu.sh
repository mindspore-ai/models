#! /bin/bash
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

if [ $# != 3 ]
then
  echo "Usage: bash scripts/run_eval_gpu.sh  DEVICE_ID /path/to/cityscapes /path/checkpoint/ENet.ckpt "
  echo "Example: bash scripts/run_eval_gpu.sh 4 /home/name/cityscapes /path/checkpoint/ENet.ckpt "
  exit 1
fi

if [ ! -d $2 ]
then 
    echo "error: DATASET_PATH=$2 is not a directory"
exit 1
fi

# Get current script path
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)

mkdir ./log_eval
cd ./log_eval

echo "DEVICE_ID: $1"
echo "cityscapes_path: $2"
echo "ckpt_path: $3"

export DEVICE_ID=$1
export RANK_SIZE=1
cityscapes_path=$2
ckpt_path=$3

python -u $BASE_PATH/../eval.py \
  --data_path ${cityscapes_path} \
  --run_distribute false \
  --encode false \
  --model_root_path ${ckpt_path} \
  --device_id 1 \
  --device_target GPU \
  > log_eval.txt 2>&1 &


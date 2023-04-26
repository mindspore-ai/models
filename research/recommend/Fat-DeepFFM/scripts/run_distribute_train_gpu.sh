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
echo "Please run the script as: "
echo "bash scripts/run_distribute_train_gpu.sh DEVICE_NUM DATASET_PATH"
echo "for example: bash scripts/run_distribute_train_gpu.sh 8 /dataset_path"
echo "After running the script, the network runs in the background, The log will be generated in log/output.log"

if [ $# != 2 ]; then
  echo "Usage: bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [DATASET_PATH]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

dataset_path=$(get_real_path $2)
echo $dataset_path

if [ ! -d $dataset_path ]
then
    echo "error: dataset_path=$dataset_path is not a directory."
exit 1
fi

export RANK_SIZE=$1
export DATA_URL=$2

rm -rf log
mkdir ./log
cp *.py ./log
cp -r ./src ./log
cd ./log || exit
env > env.log
mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
  python -u train.py \
    --dataset_path=$DATA_URL \
    --ckpt_path="Fat-DeepFFM" \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target='GPU' \
    --do_eval=True > output.log 2>&1 &
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

if [ $# != 3 ]
then
    echo "Usage: bash ./scripts/distribute_train_512p.sh [DEVICE_NUM] [DISTRIBUTE] [DATASET_PATH]"
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in log"

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

export RANK_SIZE=$1
export DISTRIBUTE=$2
export DATASET_PATH=$(get_real_path $3)

rm -rf device
mkdir device
cp -r ./src ./device
cp -r ./*.py ./device
cp ./*.yaml ./device
cd ./device
mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
                     --allow-run-as-root python train.py --run_distribute $DISTRIBUTE \
                     --data_root $DATASET_PATH --name label2city_512p_feat \
                     --instance_feat True > log 2>&1 &
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
if [ $# != 3 ]
then
    echo "Usage: bash run_standalone_train_gpu.sh [DATA_PATH] [OUTPUT_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_PATH=$(get_real_path $1)
OUTPUT_PATH=$(get_real_path $2)
export DATASET=$DATA_PATH
export OUTPUT_PATH=$OUTPUT_PATH
export DEVICE_ID=$3


rm -rf ./train
mkdir ./train
cp ../*.py ./train
cp -r ../src ./train/src
cd ./train || exit

echo "start training on DEVICE $DEVICE_ID"
echo "the results will saved in $OUTPUT_PATH"
python -u ./train.py --data_path=$DATASET --device_id=$DEVICE_ID --device_target=GPU --output_path=$OUTPUT_PATH > log 2>&1 &



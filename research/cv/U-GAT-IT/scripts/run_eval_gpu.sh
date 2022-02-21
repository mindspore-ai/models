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

if [[ $# -le 3 ]]; then
    echo "Usage: bash run_eval_gpu.sh \
    [DEVICE_ID] [DATA_PATH] [OUTPUT_PATH] \
    [LIGHT] [<INCEPTION_CHECKPOINT_PATH>]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DEVICE_ID=$1
DATA_PATH=$(get_real_path $2)
OUTPUT_PATH=$(get_real_path $3)
LIGHT=$4
INCEPTION_CHECKPOINT_PATH=''

if [ ! -d $DATA_PATH ]
then
    echo "error: DATA_PATH=$DATA_PATH is not a dir"
exit 1
fi

if [[ $# == 5 ]]; then
    INCEPTION_CHECKPOINT_PATH=$(get_real_path $5)
    if [ ! -f $INCEPTION_CHECKPOINT_PATH ]
        then
            echo "error: INCEPTION_CHECKPOINT_PATH=$INCEPTION_CHECKPOINT_PATH does not exist"
        exit 1
    fi
fi

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../src/default_config.yaml"

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
echo "start eval for device $DEVICE_ID"

if [[ $# == 5 ]]; then
    python eval.py \
    --device_target GPU \
    --device_id $DEVICE_ID \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --light $LIGHT \
    --config_path $CONFIG_FILE \
    --compute_metrics True \
    --inception_ckpt_path $INCEPTION_CHECKPOINT_PATH &> log &
else
    python eval.py \
    --device_target GPU \
    --device_id $DEVICE_ID \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --light $LIGHT \
    --config_path $CONFIG_FILE &> log &
fi

cd ..

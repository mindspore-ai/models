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


echo 'Preprocess key mapping...'
python key_mapping.py

echo 'Trans pytorch model to mindspore model.'
if [ $# -lt 4 ] ||  [ $# -gt 5 ]
then
    echo "Usage: bash torch2msp.sh [DATA_DIR] [DATA_NAME] [TORCH_PT_PATH]
     [CONFIG_PATH] [DEVICE_ID(optional)]"
exit 1
fi

export DEVICE_ID=0
if [ $# = 5 ] ; then
  export DEVICE_ID=$5
fi;

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_DIR=$(get_real_path $1)
if [ ! -d $DATA_DIR ]
then
    echo "error: DATA_DIR=$DATA_DIR is not a directory"
exit 1
fi

DATA_NAME=$2
PT_PATH=$3
CONFIG_PATH=$4

echo "DATA_DIR="$DATA_DIR
echo "DATA_NAME="$DATA_NAME
echo "PT_PATH="$PT_PATH
echo "CONFIG_PATH="$CONFIG_PATH

export CONFIG_PATH=${CONFIG_PATH}
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ ! -d "torch2msp_model" ];
then
    mkdir ./torch2msp_model
fi

echo "Start evaluation for device $DEVICE_ID :)"

python ./torch2msp/torch2msp.py \
  --device_id=$DEVICE_ID \
  --datadir=$DATA_DIR \
  --dataset=$DATA_NAME \
  --pt_path=$PT_PATH \
  --device="GPU" &> torch2msp_$DATA_NAME.log &



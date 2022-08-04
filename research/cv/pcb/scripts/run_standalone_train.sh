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


if [ $# != 4 ] && [ $# != 5 ]
then 
    echo "Usage: bash run_standalone_train.sh [MODEL_NAME] [DATASET_NAME] [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)"
exit 1
fi


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

MODEL_NAME=$1
DATASET_NAME=$2
DATASET_PATH=$(get_real_path $3)
CONFIG_PATH=$(get_real_path $4)

if [ $# == 5 ]; then
  PRETRAINED_CKPT_PATH=$(get_real_path $5)
else
  PRETRAINED_CKPT_PATH=""
fi

if [ $MODEL_NAME != "PCB" ] && [ $MODEL_NAME != "RPP" ]
then 
    echo "error: MODEL_NAME=$MODEL_NAME is invalid, please choose from ['PCB','RPP']"
exit 1
fi

if [ $DATASET_NAME != "market" ] && [ $DATASET_NAME != "duke" ] && [ $DATASET_NAME != "cuhk03" ]
then 
    echo "error: DATASET_NAME=$DATASET_NAME is invalid, please choose from ['market','duke','cuhk03']"
exit 1
fi

if [ ! -d $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi


if ( [ $MODEL_NAME = "PCB" ] && [ $DATASET_NAME = "market" ] ) || ( [ $MODEL_NAME = "PCB" ] && [ $DATASET_NAME = "duke" ] )
then
    if [ ! -f $CONFIG_PATH ]
    then 
        echo "error: CONFIG_PATH=$CONFIG_PATH is not a file"
    exit 1
    fi
else
    if [ ! -d $CONFIG_PATH ]
    then 
        echo "error: CONFIG_PATH=$CONFIG_PATH is not a directory"
    exit 1
    fi
fi

if [ $# == 5 ] && [ ! -f $PRETRAINED_CKPT_PATH ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PRETRAINED_CKPT_PATH is not a file"
exit 1
fi

export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

script_path=$(readlink -f "$0")
script_dir_path=$(dirname "${script_path}")

LOG_SAVE_PATH=${script_dir_path}/output/log/${MODEL_NAME}/${DATASET_NAME}/train/
CHECKPOINT_SAVE_PATH=${script_dir_path}/output/checkpoint/${MODEL_NAME}/${DATASET_NAME}/train/

if [ -d ${LOG_SAVE_PATH} ];
then
    rm -rf ${LOG_SAVE_PATH}
fi

if [ -d ${CHECKPOINT_SAVE_PATH} ];
then
    rm -rf ${CHECKPOINT_SAVE_PATH}
fi

if [ $MODEL_NAME = "PCB" ] && [ $DATASET_NAME = "market" ]; then
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path=$CONFIG_PATH --checkpoint_file_path=$PRETRAINED_CKPT_PATH --output_path ${script_dir_path}/output/
elif [ $MODEL_NAME = "PCB" ] && [ $DATASET_NAME = "duke" ]; then
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path=$CONFIG_PATH --checkpoint_file_path=$PRETRAINED_CKPT_PATH --output_path ${script_dir_path}/output/
elif [ $MODEL_NAME = "PCB" ] && [ $DATASET_NAME = "cuhk03" ]; then
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path ${CONFIG_PATH}/train_PCB.yaml --checkpoint_file_path=$PRETRAINED_CKPT_PATH --output_path ${script_dir_path}/output/
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path ${CONFIG_PATH}/finetune_PCB.yaml --checkpoint_file_path ${CHECKPOINT_SAVE_PATH}/PCB-40_115.ckpt --output_path ${script_dir_path}/output/
elif [ $MODEL_NAME = "RPP" ] && [ $DATASET_NAME = "market" ]; then
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path ${CONFIG_PATH}/train_PCB.yaml --checkpoint_file_path=$PRETRAINED_CKPT_PATH --output_path ${script_dir_path}/output/
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path ${CONFIG_PATH}/train_RPP.yaml --checkpoint_file_path ${CHECKPOINT_SAVE_PATH}/PCB-20_202.ckpt --output_path ${script_dir_path}/output/ 
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path ${CONFIG_PATH}/finetune_RPP.yaml --checkpoint_file_path ${CHECKPOINT_SAVE_PATH}/RPP-45_202.ckpt --output_path ${script_dir_path}/output/
elif [ $MODEL_NAME = "RPP" ] && [ $DATASET_NAME = "duke" ]; then
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path ${CONFIG_PATH}/train_PCB.yaml --checkpoint_file_path=$PRETRAINED_CKPT_PATH --output_path ${script_dir_path}/output/
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path ${CONFIG_PATH}/train_RPP.yaml --checkpoint_file_path ${CHECKPOINT_SAVE_PATH}/PCB-20_258.ckpt --output_path ${script_dir_path}/output/
else
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path ${CONFIG_PATH}/train_PCB.yaml --checkpoint_file_path=$PRETRAINED_CKPT_PATH --output_path ${script_dir_path}/output/
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path ${CONFIG_PATH}/train_RPP.yaml --checkpoint_file_path ${CHECKPOINT_SAVE_PATH}/PCB-40_115.ckpt --output_path ${script_dir_path}/output/
    python ${script_dir_path}/../train.py --dataset_path=$DATASET_PATH --config_path ${CONFIG_PATH}/finetune_RPP.yaml --checkpoint_file_path ${CHECKPOINT_SAVE_PATH}/RPP-45_115.ckpt --output_path ${script_dir_path}/output/
fi

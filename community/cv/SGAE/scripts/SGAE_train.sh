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
if [ $# -lt 2 ]
then
    echo "Usage: bash scripts/SGAE_train.sh [DATASE_FOLDER] [DATASET_NAME]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

export ASCEND_GLOBAL_LOG_LEVEL=3
export DATASET_NAME=$2
DATASE_FOLDER=$(get_real_path $1)


if [ $DATASET_NAME = "mnist" ] || [ $DATASET_NAME = "reuters" ]
then
    hidden_dim='[168, 64, 32]'
    
elif [ $DATASET_NAME = "20news" ]
then
    hidden_dim='[1024, 256, 64, 20]'
    
else
    hidden_dim='auto'
fi

if [ ! -d log  ]
then
  mkdir log
fi

if [ ! -d results  ]
then
  mkdir results
fi

if [ ! -d saved_model  ]
then
  mkdir saved_model
fi

echo  "start training for dataset $DATASET_NAME"
python -u SGAE_train.py \
    --data_path=$DATASE_FOLDER \
    --data_name=$DATASET_NAME \
    --hidden_dim=$hidden_dim > log/log_train_$DATASET_NAME.txt 2>&1 &
    

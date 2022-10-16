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
if [ $# -lt 3 ]
then
    echo "Usage: bash scripts/EAL-GAN_eval_1p.sh [DATASET_FOLDER] [DATASET_NAME] [RESUME_EPOCH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

export DATASET_NAME=$2
DATASET_FOLDER=$(get_real_path $1)
RESUME_EPOCH=$3

act_func="relu"

if [ $DATASET_NAME = "shuttle" ] || [ $DATASET_NAME = "annthyroid" ] || [ $DATASET_NAME = "mnist" ] || [ $DATASET_NAME = "attack" ]
then
    act_func="tanh"
fi

echo  "start eval for dataset $DATASET_NAME"
    
python -u TrainAndEval.py \
        --dis_activation_func=$act_func\
        --mode="eval"\
        --resume_epoch=$RESUME_EPOCH\
        --device="CPU"\
        --data_path=$DATASET_FOLDER \
        --data_name=$DATASET_NAME > log/log_val_$DATASET_NAME.txt 2>&1 &
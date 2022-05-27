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
if [ $# != 5 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "cd DBPN"
  echo "Usage: bash run_eval.sh [DEVICE_ID] [CKPT] [MODEL_TYPE] [VAL_GT_PATH] [VAL_LR_PATH]"
  echo "bash run_eval.sh 0 /data/DBPN_data/dbpn_ckpt/gen_ckpt/D-DBPN-best.ckpt DDBPN /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR"
  echo "MODE control the way of trian gan network or only train generator"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export DEVICE_ID=$1
export RANK_SIZE=1

rm -rf ./eval
mkdir ./eval
cp -r ../src ./eval
cp -r ../*.py ./eval

env > env.log
python ./eval/eval.py  --device_id=$1 --ckpt=$2 --model_type=$3 --val_GT_path=$4 --val_LR_path=$5 > eval.log 2>&1 &
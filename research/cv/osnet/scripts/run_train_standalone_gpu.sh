#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: sh run_train_standalone_ascend.sh [market1501|dukemtmcreid|cuhk03|msmt17] [DEVICE_ID] [PRETRAINED_CKPT_PATH](optional)"
exit 1
fi

if [ $1 != "market1501" ] && [ $1 != "dukemtmcreid" ] && [ $1 != "cuhk03" ] && [ $1 != "msmt17" ]
then
    echo "error: the selected dataset is not market1501, dukemtmcreid, cuhk03 or msmt17"
exit 1
fi
dataset_name=$1

if [ $# == 3 ]
then
  if [ ! -f $3 ]
  then
    echo "error: PRETRAINED_CKPT_PATH=$3 is not a file"
  exit 1
  fi
  PATH1=$(get_real_path $3)
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$2

export RANK_SIZE=1
export RANK_ID=$2

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
config_path="${BASEPATH}/../osnet_config.yaml"
echo "config path is : ${config_path}"

if [ -d "./train" ];
then
    rm -rf ./train
fi
mkdir ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
if [ $# == 2 ];
then
  python ${BASEPATH}/../train.py --config_path=$config_path --source=$dataset_name \
  --device_target="GPU" --output_path='./output'> train.log 2>&1 &
else
  python ${BASEPATH}/../train.py --config_path=$config_path --source=$dataset_name \
  --device_target="GPU" --output_path='./output' --checkpoint_file_path=$PATH1> train.log 2>&1 &
fi

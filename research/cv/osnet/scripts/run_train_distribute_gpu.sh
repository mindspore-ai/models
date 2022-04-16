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
    echo "Usage: sh run_train_distribute_ascend.sh [DEVICE_NUM] [market1501|dukemtmcreid|cuhk03|msmt17] [PRETRAINED_CKPT_PATH](optional)"
exit 1
fi

if [ $2 != "market1501" ] && [ $2 != "dukemtmcreid" ] && [ $2 != "cuhk03" ] && [ $2 != "msmt17" ]
then
    echo "error: the selected dataset is not market1501, dukemtmcreid, cuhk03 or msmt17"
exit 1
fi
dataset_name=$2

if [ $# == 3 ]
then
  if [ ! -f $3 ]
  then
    echo "error: PRETRAINED_CKPT_PATH=$3 is not a file"
  exit 1
  fi
  PATH1=$(realpath $3)
fi

ulimit -u unlimited
export RANK_SIZE=$1


BASEPATH=$(cd "`dirname $0`" || exit; pwd)
config_path="${BASEPATH}/../osnet_config.yaml"
echo "config path is : ${config_path}"
if [ -d "./train" ];
then
    rm -rf ./train
fi
mkdir ./train
cd ./train || exit
echo "start training on $RANK_SIZE device"

if [ $# == 2 ];
then
  mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root\
      python ${BASEPATH}/../train.py \
      --config_path=$config_path \
      --device_target="GPU"  \
      --run_distribute=True  \
      --source=$dataset_name \
      --output_path='./osnet_ckpt_output' > train.log 2>&1 &
else
  mpirun -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root\
      python ${BASEPATH}/../train.py \
      --config_path=$config_path \
      --device_target="GPU"  \
      --run_distribute=True  \
      --source=$dataset_name \
      --output_path='./osnet_ckpt_output' \
      --checkpoint_file_path=$PATH1> train.log 2>&1 &
fi


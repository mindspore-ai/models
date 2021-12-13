#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

if [ $# != 3 ]
then
    echo "Usage: bash run_eval_ascend.sh [market1501|dukemtmcreid|cuhk03|msmt17] [CHECKPOINT_PATH] [DEVICE_ID]"
exit 1
fi

if [ $1 != "market1501" ] && [ $1 != "dukemtmcreid" ] && [ $1 != "cuhk03" ] && [ $1 != "msmt17" ]
then
    echo "error: the selected dataset is not market1501, dukemtmcreid, cuhk03 or msmt17"
exit 1
fi
dataset_name=$1


if [ ! -f $2 ]
then
  echo "error: CHECKPOINT_PATH=$2 is not a file"
exit 1
fi
PATH1=$(realpath $2)

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$3

export RANK_SIZE=1
export RANK_ID=$3

config_path="${BASEPATH}/../osnet_config.yaml"
echo "config path is : ${config_path}"

if [ -d "./eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cd ./eval || exit
echo "start evaluating for device $DEVICE_ID"
python ${BASEPATH}/../eval.py --config_path=$config_path --checkpoint_file_path=$PATH1 --target=$dataset_name> ./eval.log 2>&1 &

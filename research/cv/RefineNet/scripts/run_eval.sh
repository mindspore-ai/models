#! /bin/bash
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
if [ $# -ne 4 ]
then 
    echo "Usage: bash scripts/run_eval_ascend.sh [DATA_LST] [PRETRAINED_PATH] [DEVICE_TARGET] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $1)
PRETRAINED_PATH=$(get_real_path $2)
echo $DATASET_PATH
echo $PRETRAINED_PATH

if [ ! -f $DATASET_PATH ]
then 
    echo "error: DATASET_PATH=$DATASET_PATH is not a file"
exit 1
fi

if [ ! -f $PRETRAINED_PATH ]
then 
    echo "error: PRETRAINED_PATH=$PRETRAINED_PATH is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_TARGET=$3
export DEVICE_ID=$4
export RANK_ID=0
export RANK_SIZE=1
LOCAL_DIR=eval$DEVICE_ID
rm -rf $LOCAL_DIR
mkdir $LOCAL_DIR
cp ../*.py $LOCAL_DIR
cp *.sh $LOCAL_DIR
cp -r ../src $LOCAL_DIR
cd $LOCAL_DIR || exit
echo "start training for device $DEVICE_ID"
env > env.log
python eval.py --data_lst=$DATASET_PATH --ckpt_path=$PRETRAINED_PATH --device_target=$DEVICE_TARGET --device_id=$DEVICE_ID --flip &> log &
cd ..


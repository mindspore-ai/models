#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: bash scripts/run_standalone_train.sh [DATASET_NAME] [DATASET_PATH] [DEVICE_ID] [PLATFORM](optional) [RESUME_CKPT](optional for resume)"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_NAME=$1
PATH1=$(get_real_path $2)
export DEVICE_ID=$3
if [ $# == 4 ]; then
  PLATFORM=$4
else
  PLATFORM="Ascend"
fi

if [ ! -d $PATH1 ]; then
  echo "error: DATASET_PATH=$PATH1 is not a directory"
  exit 1
fi

if [ $# == 5 ]; then
  RESUME_CKPT=$(get_real_path $5)
  if [ ! -f $RESUME_CKPT ]; then
    echo "error: RESUME_CKPT=$RESUME_CKPT is not a file"
    exit 1
  fi
  echo $RESUME_CKPT
fi

run_ascend() {
  ulimit -u unlimited
  export DEVICE_NUM=1
  export RANK_ID=0
  export RANK_SIZE=1

  echo "start training for device $DEVICE_ID"
  env >env.log
  python train.py --train_dataset=$DATASET_NAME --train_dataset_path=$PATH1 --device_target=Ascend \
                  --resume_ckpt=$RESUME_CKPT > log.txt 2>&1 &
  cd ..
}

run_gpu() {
  env >env.log
  python train.py --train_dataset=$DATASET_NAME --train_dataset_path=$PATH1 --device_target=GPU \
                  --resume_ckpt=$RESUME_CKPT > log.txt 2>&1 &
  cd ..
}

if [ -d "train" ]; then
    rm -rf ./train
fi
WORKDIR=./train${DEVICE_ID}
rm -rf $WORKDIR
mkdir $WORKDIR
cp ./*.py $WORKDIR
cp -r ./src $WORKDIR
cp ./*yaml $WORKDIR
cd $WORKDIR || exit

if [ "Ascend" == $PLATFORM ]; then
  run_ascend $PATH1
elif [ "GPU" == $PLATFORM ]; then
  run_gpu $PATH1
else
  echo "error: PLATFORM=$PLATFORM is not support, only support Ascend and GPU."
fi

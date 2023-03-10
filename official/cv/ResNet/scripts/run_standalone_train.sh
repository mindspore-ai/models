#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

CURPATH="$(dirname "$0")"
# shellcheck source=/dev/null
. ${CURPATH}/cache_util.sh

if [ $# != 2 ] && [ $# != 3 ] && [ $# != 4 ] && [ $# != 5 ]
then
    echo "Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH]"
    echo "Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RESUME_CKPT](optional)"
    echo "Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional)"
    echo "Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [RUN_EVAL](optional) [EVAL_DATASET_PATH](optional) [RESUME_CKPT](optional)"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
CONFIG_FILE=$(get_real_path $2)
if [ $# == 3 ]
then
    RESUME_CKPT=$(get_real_path $3)
fi

if [ $# == 4 ]
then
  RUN_EVAL=$3
  EVAL_DATASET_PATH=$(get_real_path $4)
fi

if [ $# == 5 ]
then
  RUN_EVAL=$3
  EVAL_DATASET_PATH=$(get_real_path $4)
  RESUME_CKPT=$(get_real_path $5)
fi

if [ ! -d $PATH1 ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

if [ $# == 3 ] && [ ! -f $RESUME_CKPT ]
then
    echo "error: RESUME_CKPT=$RESUME_CKPT is not a file"
exit 1
fi

if [ "x${RUN_EVAL}" == "xTrue" ] && [ ! -d $EVAL_DATASET_PATH ]
then
  echo "error: EVAL_DATASET_PATH=$EVAL_DATASET_PATH is not a directory"
  exit 1
fi

if [ "x${RUN_EVAL}" == "xTrue" ]
then
  bootup_cache_server
  CACHE_SESSION_ID=$(generate_cache_session)
fi

export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../config/*.yaml ./train
cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
echo "start training"
env > env.log
if [ $# == 2 ]
then
    python train.py  --data_path=$PATH1 --config_path=$CONFIG_FILE --output_dir '../outputs' &> log.txt &
fi

if [ $# == 3 ]
then
    python train.py --data_path=$PATH1 --resume_ckpt=$RESUME_CKPT --config_path=$CONFIG_FILE --output_dir '../outputs' &> log.txt &
fi

if [ $# == 4 ]
then
    python train.py --data_path=$PATH1 --run_eval=$RUN_EVAL --eval_dataset_path=$EVAL_DATASET_PATH \
           --enable_cache=True --cache_session_id=$CACHE_SESSION_ID \
           --config_path=$CONFIG_FILE --output_dir '../outputs' &> log.txt &
    if [ "x${RUN_EVAL}" == "xTrue" ]
    then
      echo -e "\nWhen training run is done, remember to shut down the cache server via \"cache_admin --stop\""
    fi
fi

if [ $# == 5 ]
then
    python train.py --data_path=$PATH1 --run_eval=$RUN_EVAL --eval_dataset_path=$EVAL_DATASET_PATH \
           --enable_cache=True --cache_session_id=$CACHE_SESSION_ID --resume_ckpt=$RESUME_CKPT \
           --config_path=$CONFIG_FILE --output_dir '../outputs' &> log.txt &
    if [ "x${RUN_EVAL}" == "xTrue" ]
    then
      echo -e "\nWhen training run is done, remember to shut down the cache server via \"cache_admin --stop\""
    fi
fi
cd ..

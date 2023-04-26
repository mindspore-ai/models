#!/usr/bin/env bash
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

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

run_ascend()
{
    # check pretrain_ckpt file
    if [ ! -f $4 ]
    then
        echo "error: PRETRAIN_CKPT=$4 is not a file"
    exit 1
    fi

    # set environment
    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    CONFIG_FILE=$(get_real_path $2)
    DATASET_PATH=$(get_real_path $3)
    CKPT_PATH=$(get_real_path $4)
    export PYTHONPATH=${BASEPATH}:$PYTHONPATH
    export DEVICE_ID=0
    export RANK_ID=0
    export RANK_SIZE=1
    if [ -d "../eval" ];
    then
        rm -rf ../eval
    fi
    mkdir ../eval
    cd ../eval || exit

    # launch
    python ${BASEPATH}/../eval.py \
            --config_path=$CONFIG_FILE \
            --platform=$1 \
            --dataset_path=$DATASET_PATH \
            --pretrain_ckpt=$CKPT_PATH \
            &> ../eval.log &  # dataset val folder path
}

run_gpu()
{
    # check pretrain_ckpt file
    if [ ! -f $4 ]
    then
        echo "error: PRETRAIN_CKPT=$4 is not a file"
    exit 1
    fi

    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    CONFIG_FILE=$(get_real_path $2)
    DATASET_PATH=$(get_real_path $3)
    CKPT_PATH=$(get_real_path $4)
    export PYTHONPATH=${BASEPATH}:$PYTHONPATH
    if [ -d "../eval" ];
    then
        rm -rf ../eval
    fi
    mkdir ../eval
    cd ../eval || exit

    python ${BASEPATH}/../eval.py \
        --config_path=$CONFIG_FILE \
        --platform=$1 \
        --dataset_path=$DATASET_PATH \
        --pretrain_ckpt=$CKPT_PATH \
        &> ../eval.log &  # dataset train folder
}

run_cpu()
{
    # check pretrain_ckpt file
    if [ ! -f $4 ]
    then
        echo "error: PRETRAIN_CKPT=$4 is not a file"
    exit 1
    fi

    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    CONFIG_FILE=$(get_real_path $2)
    DATASET_PATH=$(get_real_path $3)
    CKPT_PATH=$(get_real_path $4)
    export PYTHONPATH=${BASEPATH}:$PYTHONPATH
    if [ -d "../eval" ];
    then
        rm -rf ../eval
    fi
    mkdir ../eval
    cd ../eval || exit

    python ${BASEPATH}/../eval.py \
        --config_path=$CONFIG_FILE \
        --platform=$1 \
        --dataset_path=$DATASET_PATH \
        --pretrain_ckpt=$CKPT_PATH \
        &> ../eval.log &  # dataset train folder
}


if [ $# -ne 4 ]
then
    echo "Usage:
          Ascend: bash run_eval.sh [PLATFORM] [CONFIG_PATH] [DATASET_PATH] [PRETRAIN_CKPT]
          GPU: bash run_eval.sh [PLATFORM] [CONFIG_PATH] [DATASET_PATH] [PRETRAIN_CKPT]
          CPU: bash run_eval.sh [PLATFORM] [CONFIG_PATH] [DATASET_PATH] [PRETRAIN_CKPT]"
exit 1
fi

# check dataset path
if [ ! -d $3 ]
then
    echo "error: DATASET_PATH=$3 is not a directory"
exit 1
fi

if [ $1 = "CPU" ] ; then
    run_cpu "$@"
elif [ $1 = "GPU" ] ; then
    run_gpu "$@"
elif [ $1 = "Ascend" ] ; then
    run_ascend "$@"
else
    echo "Unsupported platform."
fi;

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

run_ascend_or_gpu()
{
    if [ $# = 5 ] ; then
        PRETRAINED_CKPT=""
        FREEZE_LAYER="none"
        FILTER_HEAD="False"
    elif [ $# = 7 ] ; then
        PRETRAINED_CKPT=$6
        FREEZE_LAYER=$7
        FILTER_HEAD="False"
    elif [ $# = 8 ] ; then
        PRETRAINED_CKPT=$6
        FREEZE_LAYER=$7
        FILTER_HEAD=$8
    else
        echo "Usage:
              Ascend: bash run_train.sh $1 [CONFIG_PATH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH] [CKPT_PATH](optional) [FREEZE_LAYER](optional) [FILTER_HEAD](optional)
              Ascend: bash run_train.sh $1 [CONFIG_PATH] [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]"
        exit 1
    fi;

    if [ $3 -lt 1 ] || [ $3 -gt 8 ]
    then
        echo "error: DEVICE_NUM=$3 is not in (1-8)"
    exit 1
    fi

    if [ ! -d $5 ] && [ ! -f $5 ]
    then
        echo "error: DATASET_PATH=$5 is not a directory or file"
    exit 1
    fi

    get_real_path(){
        if [ "${1:0:1}" == "/" ]; then
            echo "$1"
        else
            echo "$(realpath -m $PWD/$1)"
        fi
    }
    
    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    CONFIG_FILE=$(get_real_path $2)
    DATASET_PATH=$(get_real_path $5)
    
    VISIABLE_DEVICES=$4
    IFS="," read -r -a CANDIDATE_DEVICE <<< "$VISIABLE_DEVICES"
    if [ ${#CANDIDATE_DEVICE[@]} -ne $3 ]
    then
        echo "error: DEVICE_NUM=$3 is not equal to the length of VISIABLE_DEVICES=$4"
    exit 1
    fi

    export PYTHONPATH=${BASEPATH}:$PYTHONPATH
    if [ $1 = "Ascend" ] && [ $3 -eq 1 ]; then
        export DEVICE_ID=${CANDIDATE_DEVICE[0]}
        export RANK_ID=0
    elif [ $1 = "GPU" ]; then
        export CUDA_VISIBLE_DEVICES="$4"
    fi

    if [ -d "../train" ];
    then
        rm -rf ../train
    fi
    mkdir ../train
    cd ../train || exit

    RUN_DISTRIBUTE=True
    if [ $3 -eq 1 ] ; then
        RUN_DISTRIBUTE=False
        nohup python ${BASEPATH}/../train.py \
            --run_distribute=$RUN_DISTRIBUTE \
            --config_path=$CONFIG_FILE \
            --platform=$1 \
            --dataset_path=$DATASET_PATH \
            --pretrain_ckpt=$PRETRAINED_CKPT \
            --freeze_layer=$FREEZE_LAYER \
            --filter_head=$FILTER_HEAD \
            &> ../train.log &
        exit 1
    fi

    # mpirun for multi card
    mpirun -n $3 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python ${BASEPATH}/../train.py \
        --config_path=$CONFIG_FILE \
        --platform=$1 \
        --run_distribute=$RUN_DISTRIBUTE \
        --dataset_path=$DATASET_PATH \
        --pretrain_ckpt=$PRETRAINED_CKPT \
        --freeze_layer=$FREEZE_LAYER \
        --filter_head=$FILTER_HEAD \
        &> ../train.log &  # dataset train folder

}

run_cpu()
{
    if [ $# = 3 ] ; then
        PRETRAINED_CKPT=""
        FREEZE_LAYER="none"
        FILTER_HEAD="False"
    elif [ $# = 5 ] ; then
        PRETRAINED_CKPT=$4
        FREEZE_LAYER=$5
        FILTER_HEAD="False"
    elif [ $# = 6 ] ; then
        PRETRAINED_CKPT=$4
        FREEZE_LAYER=$5
        FILTER_HEAD=$6
    else
        echo "Usage:
              CPU: bash run_train.sh CPU [CONFIG_PATH] [DATASET_PATH]
              CPU: bash run_train.sh CPU [CONFIG_PATH] [DATASET_PATH] [CKPT_PATH](optional) [FREEZE_LAYER](optional) [FILTER_HEAD](optional)"
        exit 1
    fi;
    if [ ! -d $3 ]
    then
        echo "error: DATASET_PATH=$3 is not a directory"
    exit 1
    fi

    get_real_path(){
        if [ "${1:0:1}" == "/" ]; then
            echo "$1"
        else
            echo "$(realpath -m $PWD/$1)"
        fi
    }

    BASEPATH=$(cd "`dirname $0`" || exit; pwd)
    CONFIG_FILE=$(get_real_path $2)
    DATASET_PATH=$(get_real_path $3)

    export PYTHONPATH=${BASEPATH}:$PYTHONPATH
    if [ -d "../train" ];
    then
        rm -rf ../train
    fi
    mkdir ../train
    cd ../train || exit

    python ${BASEPATH}/../train.py \
        --config_path=$CONFIG_FILE \
        --platform=$1 \
        --dataset_path=$DATASET_PATH \
        --pretrain_ckpt=$PRETRAINED_CKPT \
        --freeze_layer=$FREEZE_LAYER \
        --filter_head=$FILTER_HEAD \
        &> ../train.log &  # dataset train folder
}

if [ $1 = "Ascend" ] || [ $1 = "GPU" ] ; then
    run_ascend_or_gpu "$@"
elif [ $1 = "CPU" ] ; then
    run_cpu "$@"
else
    echo "Unsupported platform."
fi;

#!/bin/bash

if [ $# != 2 ]
then
    echo "Usage: sh scripts/run_train.sh [device_num][RANK_TABLE_FILE]"
exit 1
fi

if [ ! -f $2 ]
then
    echo "error: RANK_TABLE_FILE=$2 is not a file"
exit 1
fi

export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export HCCL_CONNECT_TIMEOUT=600
export GLOG_v=1

ulimit -u unlimited
ulimit -SHn 65535
export DEVICE_NUM=$1
export RANK_SIZE=$1
RANK_TABLE_FILE=$(realpath $2)
export RANK_TABLE_FILE
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
for((i=0; i<$1; i++));
do
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    cd ./train_parallel$i ||exit
    env > env.log
    python ../../pretrain_three_ms.py \
        --config ../../config/pretrain_three_modal_txt_img_audio_config.json \
        --use_parallel "true"  > log 2>&1 &
    cd ..
done

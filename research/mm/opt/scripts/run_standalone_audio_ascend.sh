#!/bin/bash
ulimit -SHn 65535
rm -rf ./train_alone_audio
mkdir ./train_alone_audio
cd ./train_alone_audio
python3 ../../pretrain_three_audio_local.py \
    --config ../../config/pretrain_three_modal_audio_local_config.json \
    --use_parallel False > log 2>&1 &
cd ..

export DEVICE_ID=7 && \
python3 pretrain_three_audio_local.py \
--config config/pretrain_three_modal_audio_local_config.json \
--use_parallel False


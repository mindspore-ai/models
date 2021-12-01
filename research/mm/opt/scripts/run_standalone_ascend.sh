#!/bin/bash
ulimit -SHn 65535
rm -rf ./train_alone
mkdir ./train_alone
cd ./train_alone
python3 ../../pretrain_three_ms.py \
    --config ../../config/pretrain_three_modal_txt_img_audio_config.json \
    --use_parallel False > log 2>&1 &
cd ..


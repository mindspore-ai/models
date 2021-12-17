#!/bin/bash

export RANK_SIZE=1;export DEVICE_ID=0;python pretrain_three_caption_local.py \
    --config=./config/ftcap_coco_bu_zh_lr5e-5.json \
    --output_dir=./output/ftcap_coco_bu_zh_lr5e-5 \
    --use_parallel=False \
    --data_url="a" --train_url="a" --audio_dim=512
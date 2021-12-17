#!/bin/bash

export RANK_SIZE=1;export DEVICE_ID=0;python pretrain_three_caption_eval.py \
    --config=./config/ftcap_coco_bu_zh_lr5e-5.json \
    --output_dir=./output/ftcap_coco_bu_zh_lr5e-5 \
    --use_parallel=False \
    --ckpt_file "./output/ftcap_coco_bu_zh_lr5e-5/ckpt/rank_0/OPT_caption-3875_40.ckpt" \
    --data_url="a" --train_url="a" --audio_dim=512
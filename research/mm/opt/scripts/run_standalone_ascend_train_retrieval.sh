#!/bin/bash
export GLOG_v=2;export RANK_SIZE=1;export DEVICE_ID=3;python finetune_itm_three.py \
--config=./config/test_ch_retft.json \
--use_parallel=False \
--data_url="a" --train_url="a"
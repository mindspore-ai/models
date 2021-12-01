#!/bin/bash
export GLOG_v=2;export RANK_SIZE=1;export DEVICE_ID=3;python pretrain_three_retrieval.py \
--config=./config/test_ch.json \
--use_parallel=False \
--data_url="a" --train_url="a"
#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

if [ $# != 2 ]
then
    echo "Usage: bash run_standalone_train_gpu.sh [DATASET] [DEVICE_ID]"
exit 1
fi

DATASET=$1
DEVICE_ID=$2

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export GLOG_v=3

echo "DATASET: $DATASET     DEVICE_ID: $DEVICE_ID"

if [ ! -d "checkpoints" ]
then
    mkdir checkpoints
fi

if [ ! -d "logs" ]
then
    mkdir logs
fi

echo "Start training..."
if [ $DATASET == reuters16 ]
then
    nohup python -u train.py $DATASET data/reuters.tfidf.mat --num_features 16 --num_neighbors 11 --batch_size 128 --lr 0.003 \
    --num_trees 13 --temperature 0.7 --alpha 0.1 --beta 0.09 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == reuters32 ]
then
    nohup python -u train.py $DATASET data/reuters.tfidf.mat --num_features 32 --num_neighbors 15 --batch_size 128 --lr 0.003 \
    --num_trees 10 --temperature 0.5 --alpha 0.3 --beta 0.03 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == reuters64 ]
then
    nohup python -u train.py $DATASET data/reuters.tfidf.mat --num_features 64 --num_neighbors 20 --batch_size 32 --lr 0.0005 \
    --num_trees 19 --temperature 0.2 --alpha 0.2 --beta 0.05 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == reuters128 ]
then
    nohup python -u train.py $DATASET data/reuters.tfidf.mat --num_features 128 --num_neighbors 20 --batch_size 64 --lr 0.0005 \
    --num_trees 14 --temperature 0.3 --alpha 0.2 --beta 0.04 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == tmc16 ]
then
    nohup python -u train.py $DATASET data/tmc.tfidf.mat --num_features 16 --num_neighbors 10 --batch_size 32 --lr 0.001 \
    --num_trees 10 --temperature 0.8 --alpha 0.2 --beta 0.2 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == tmc32 ]
then
    nohup python -u train.py $DATASET data/tmc.tfidf.mat --num_features 32 --num_neighbors 8 --batch_size 128 --lr 0.001 \
    --num_trees 19 --temperature 0.3 --alpha 0.1 --beta 0.1 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == tmc64 ]
then
    nohup python -u train.py $DATASET data/tmc.tfidf.mat --num_features 64 --num_neighbors 3 --batch_size 64 --lr 0.0005 \
    --num_trees 19 --temperature 0.1 --alpha 0.2 --beta 0.08 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == tmc128 ]
then
    nohup python -u train.py $DATASET data/tmc.tfidf.mat --num_features 128 --num_neighbors 7 --batch_size 64 --lr 0.0005 \
    --num_trees 10 --temperature 0.1 --alpha 0.7 --beta 0.03 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == ng16 ]
then
    nohup python -u train.py $DATASET data/ng20.tfidf.mat --num_features 16 --num_neighbors 7 --batch_size 128 --lr 0.001 \
    --num_trees 17 --temperature 0.3 --alpha 0.2 --beta 0.1 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == ng32 ]
then
    nohup python -u train.py $DATASET data/ng20.tfidf.mat --num_features 32 --num_neighbors 10 --batch_size 128 --lr 0.001 \
    --num_trees 20 --temperature 0.1 --alpha 0.1 --beta 0.1 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == ng64 ]
then
    nohup python -u train.py $DATASET data/ng20.tfidf.mat --num_features 64 --num_neighbors 15 --batch_size 128 --lr 0.001 \
    --num_trees 13 --temperature 0.1 --alpha 0.1 --beta 0.05 --device GPU > logs/train_$DATASET.log 2>&1 &
elif [ $DATASET == ng128 ]
then
    nohup python -u train.py $DATASET data/ng20.tfidf.mat --num_features 128 --num_neighbors 7 --batch_size 128 --lr 0.0005 \
    --num_trees 18 --temperature 0.1 --alpha 0.3 --beta 0.05 --device GPU > logs/train_$DATASET.log 2>&1 &
fi
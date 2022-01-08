#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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
if [ $# != 3 ]
then
    echo "Usage: sh run_distribute_train_gpu.sh [CONTENT_PATH] [STYLE_PATH] [CKPT_PATH]"
    exit 1
fi

CONTENT_PATH=$1
STYLE_PATH=$2
CKPT_PATH=$3

if [ ! -d $1 ]
then
    echo "error: folder CONTENT_PATH=$1 does not exist"
exit 1
fi

if [ ! -d $2 ]
then
    echo "error: folder STYLE_PATH=$2 does not exist"
exit 1
fi

if [ ! -d $3 ]
then
    echo "error: folder CKPT_PATH=$3 does not exist"
exit 1
fi

mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root python train.py --platform GPU \
            --run_distribute 1 --device_num 8 --run_offline 1 --content_path $CONTENT_PATH \
            --style_path $STYLE_PATH --ckpt_path=$CKPT_PATH  > distribute_train_gpu_log 2>&1 &
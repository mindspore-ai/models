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
# ============================================================================

if [ $# != 6 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash ./scripts/run_eval.sh [PLATFORM] [DEVICE_ID] [CONTENT_PATH] [STYLE_PATH] [INCEPTION_CKPT] [CKPT_PATH]"
    echo "for example: bash ./scripts/run_eval.sh GPU 0 ./dataset/test/content ./dataset/test/style ./pretrained_model/inceptionv3.ckpt ./ckpt/style_transfer_rank_00_model_0060.ckpt
"
    echo "=============================================================================================================="
exit 1
fi

export PLATFORM=$1
export DEVICE_ID=$2
export CONTENT_PATH=$3
export STYLE_PATH=$4
export INCEPTION_CKPT=$5
export CKPT_PATH=$6

if [ ! -d $3 ]
then
    echo "error: folder CONTENT_PATH=$3 does not exist"
exit 1
fi

if [ ! -d $4 ]
then
    echo "error: folder STYLE_PATH=$4 does not exist"
exit 1
fi

if [ ! -f $5 ]
then
    echo "error: file INCEPTION_CKPT=$5 does not exist"
exit 1
fi

if [ ! -f $6 ]
then
    echo "error: file CKPT_PATH=$6 does not exist"
exit 1
fi


python -u eval.py \
  --platform $PLATFORM \
  --device_id $DEVICE_ID \
  --content_path $CONTENT_PATH \
  --style_path $STYLE_PATH \
  --inception_ckpt $INCEPTION_CKPT \
  --ckpt_path $CKPT_PATH > eval_log 2>&1 &


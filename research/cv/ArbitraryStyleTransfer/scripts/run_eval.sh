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


if [ $# != 5 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash ./scripts/run_eval.sh [DEVICE_ID] [CONTENT_PATH] [STYLE_PATH] [INCEPTION_CKPT] [CKPT_PATH]"
    echo "for example: bash scripts/run_eval.sh 0 /home/style_dataset/content_test/ /home/style_dataset/style_test/ /root/ArbitraryStyleTransfer3/pretrained_model/inceptionv3.ckpt /root/ArbitraryStyleTransfer_pr/train_parallel0/ckpt/style_transfer_model_0005.ckpt
"
    echo "=============================================================================================================="
exit 1
fi

export DEVICE_ID=$1
export CONTENT_PATH=$2
export STYLE_PATH=$3
export INCEPTION_CKPT=$4
export CKPT_PATH=$5


get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}


python eval.py \
  --device_id $DEVICE_ID \
  --content_path $CONTENT_PATH \
  --style_path $STYLE_PATH \
  --inception_ckpt $INCEPTION_CKPT \
  --ckpt_path $CKPT_PATH > eval_log 2>&1 &

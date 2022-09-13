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

ulimit -m unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

chmod +x kitti/kitti_eval/evaluate_object_3d_offline

if [ ! -d dataset/KITTI/object/training/label_2/ ]; then
    echo "error: KITTI dataset label is not found"
    echo "please ensure that the KITTI dataset is downloaded and placed in the correct path"
    echo "dataset/KITTI/object/training/label_2/"
fi

if [ $# != 1 ] && [ $# != 2 ]
then
    echo "Usage: bash scripts/run_eval_gpu.sh [OUTPUT_PATH] [PRETRAINDE_CKPT]"
    echo "============================================================"
    echo "[OUTPUT_PATH]: The path to the output for kitti eval."
    echo "[PRETRAINDE_CKPT]: The path to the checkpoint file."
    echo "============================================================"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# -ge 1 ]
then
    OUTPUT_PATH=$1
    mkdir -p $OUTPUT_PATH

    if [ ! -d $OUTPUT_PATH ]
    then
        echo "error: OUTPUT_PATH=$OUTPUT_PATH is not a directory"
    exit 1
    fi
fi

if [ $# -ge 2 ]
then
    PRETRAINED_CKPT=$2
    if [ ! -f $PRETRAINED_CKPT ]
    then
        echo "error: PRETRAINED_CKPT=$PRETRAINED_CKPT is not a file"
    exit 1
    fi
fi

echo 'running test'

python eval.py --output=$OUTPUT_PATH --model_path=$PRETRAINED_CKPT > "Fpointnet_eval.log" && kitti/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ $OUTPUT_PATH  >> "Fpointnet_eval.log" 2>&1 &

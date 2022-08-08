#!/usr/bin/env bash
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
if [ $# != 4 ] ; then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_standalone_eval.sh CFG DATA_DIR CHECKPOINT_FILE_PATH DEVICE_ID"
    echo "for example: bash scripts/run_standalone_eval.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava checkpoint_epoch_00020_best248.pyth.ckpt 1"
    echo "=============================================================================================================="
    exit 1;
fi
get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
export DEVICE_ID=$4
folder=eval$DEVICE_ID
echo $cpu_range
echo "$(get_real_path $3)"
CFG=$(get_real_path $1)
DATA_DIR=$(get_real_path $2)
CHECKPOINT_FILE_PATH=$(get_real_path $3)
echo "start evaluating with device $DEVICE_ID"
cpu_range="$((DEVICE_ID*24))-$(((DEVICE_ID+1)*24-1))"
rm -rf ./$folder
mkdir ./$folder
cp -r configs ./$folder
cp -r src ./$folder
cp *.py ./$folder
cd ./$folder ||exit
env > env.log
taskset -c $cpu_range python -u eval.py --cfg "${CFG}" \
     AVA.FRAME_DIR "${DATA_DIR}/frames" \
     AVA.FRAME_LIST_DIR "${DATA_DIR}/ava_annotations" \
     AVA.ANNOTATION_DIR "${DATA_DIR}/ava_annotations" \
    TEST.CHECKPOINT_FILE_PATH "${CHECKPOINT_FILE_PATH}" > log_eval_ascend 2>&1 &
cd ..

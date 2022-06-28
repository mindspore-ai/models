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
if [ $# != 3 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_standalone_train.sh CFG DATA_DIR CHECKPOINT_FILE_PATH"
echo "for example: bash scripts/run_standalone_train.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt"
echo "=============================================================================================================="
exit 1;
fi
CFG=$(realpath $1)
DATA_DIR=$(realpath $2)
CHECKPOINT_FILE_PATH=$(realpath $3)
python -u train.py --cfg "$CFG" \
     AVA.FRAME_DIR "${DATA_DIR}/frames" \
     AVA.FRAME_LIST_DIR "${DATA_DIR}/ava_annotations" \
     AVA.ANNOTATION_DIR "${DATA_DIR}/ava_annotations" \
     TRAIN.CHECKPOINT_FILE_PATH "$CHECKPOINT_FILE_PATH" > log_standalone_ascend 2>&1 &

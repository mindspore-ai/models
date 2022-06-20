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

set -e
if [ $# != 4 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash ./scripts/run_eval.sh [device_id] [eval_epoch] [load_ckpt] [DATA_ROOT]"
    echo "For example: bash ./scripts/run_eval.sh 0 200 ./checkpoints/netG_epoch_200.ckpt ./ADEChallengeData2016"
    echo "================================================================================================================="
exit 1
fi

# Get absolute path
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

# Get current script path
DEVICE_ID=$1
EVAL_EPOCH=$2
LOAD_CKPT=$(get_real_path $3)
DATA_ROOT=$(get_real_path $4)


CUDA_VISIBLE_DEVICES=$DEVICE_ID mpirun --allow-run-as-root -n 1 \
python -u ./eval.py --distribute True \
                    --dataroot $DATA_ROOT \
                    --ckpt_dir $LOAD_CKPT \
                    --which_epoch $EVAL_EPOCH > fid_$EVAL_EPOCH.log 2>&1 &

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
# ===========================================================================
if [ $# != 7 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_eval.sh [DEVICE_ID] [MODEL_NAME ] [EVAL_BATCH_SIZE] [DATA_ROOT] [CKPT_PATH] [CKPT_NAME] [OUTPUT_PATH]"
  echo "For example:"
  echo "bash scripts/run_eval.sh 0 DAM_ubuntu 256 ABSOLUTE_PATH/data_test.mindrecord ABSOLUTE_PATH/output/ubuntu
                                ckptfille.ckpt /output/ubuntu/"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in ./eval.log"

ulimit -c unlimited
export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=$1
echo "start evaluating for device $DEVICE_ID"
python eval.py --model_name=$2 \
               --parallel=False \
               --eval_batch_size=$3 \
               --data_root=$4 \
               --ckpt_path=$5 \
               --ckpt_name=$6 \
               --output_path=$7 >eval.log 2>&1 &
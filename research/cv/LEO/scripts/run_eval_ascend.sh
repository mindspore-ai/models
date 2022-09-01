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
if [[ $# -ne 6 ]]; then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_eval_gpu.sh [DEVICE_ID] [DEVICE_TARGET] [DATA_PATH] [DATA_NAME] [NUM_TR_EXAMPLES_PER_CLASS] [CKPT_FILE] "
    echo "For example: bash scripts/run_eval_ascend.sh 4 Ascend ../leo/leo-mindspore/embeddings miniImageNet 5 ./ckpt/1P_mini_5/xxx.ckpt "
    echo "=============================================================================================================="
    exit 1;
fi

export  GLOG_v=3
export  DEVICE_ID=$1
export  DEVICE_TARGET=$2
export  DATA_PATH=$3
export  DATA_NAME=$4
export  NUM_TR_EXAMPLES_PER_CLASS=$5
export  CKPT_FILE=$6

nohup python eval.py --device_target $DEVICE_TARGET \
                     --data_path $DATA_PATH \
                     --dataset_name $DATA_NAME \
                     --num_tr_examples_per_class $NUM_TR_EXAMPLES_PER_CLASS \
                     --ckpt_file $CKPT_FILE  > ${DATA_NAME}_${NUM_TR_EXAMPLES_PER_CLASS}_eval.log 2>&1 &

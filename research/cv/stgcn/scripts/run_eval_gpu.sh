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

if [ $# != 5 ]
then
    echo "Usage: bash scripts/run_eval_gpu.sh [DATA_PATH] [CKPT_PATH] [N_PRED] [GRAPH_CONV_TYPE] [DEVICE_ID]"
exit 1
fi

export DATA_PATH=$1
export CKPT_PATH=$2
export N_PRED=$3
export GRAPH_CONV_TYPE=$4
export DEVICE_ID=$5

python eval.py --data_url=$DATA_PATH \
                --device_target="GPU"  \
                --train_url=./checkpoint \
                --run_distribute=False \
                --run_modelarts=False \
                --device_id=$DEVICE_ID \
                --ckpt_url=$CKPT_PATH \
                --n_pred=$N_PRED \
                --graph_conv_type=$GRAPH_CONV_TYPE > eval.log 2>&1 &

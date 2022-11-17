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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_train_gpu.sh"
echo "______________________________________________________________________________________________________________"
echo "If your data_path or device_id or device_target is different from those in default_config.yaml, "
echo "please run the script as: "
echo "bash run_train_gpu.sh DATA_PATH DEVICE_ID DEVICE_TARGET EVAL_WHILE_TRAIN "
echo "for example: bash ./script/run_train_gpu.sh './data/mindrecord/' 1 GPU True"
echo "         "
echo "**** FYI: only DEVICE_TARGET=GPU is supported currently. ****"
echo "______________________________________________________________________________________________________________"
echo "Then you can find detailed log and results in files: train.log, eval.log and loss.log. "
echo "         "
echo "If you want to set up more parameters by yourself, "
echo "you are suggested to check the file default_config.yaml and change parameter values there. "
echo "=============================================================================================================="

if [ $# != 4 ]
then
  echo "Usage: bash run_train_gpu.sh [DATA_PATH] [DEVICE_ID] [DEVICE_TARGET] [EVAL_WHILE_TRAIN] "
exit 1
fi

DATA_PATH=$1
DEVICE_ID=$2
DEVICE_TARGET=$3
EVAL_WHILE_TRAIN=$4

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
python ./train.py --data_path=$DATA_PATH --device_target=$DEVICE_TARGET --eval_while_train=$EVAL_WHILE_TRAIN > train.log 2>&1 &

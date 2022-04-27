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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_single_eval.sh DEVICE_ID CONFIG_PATH"
echo "for example: sh run_single_eval.sh 0 home/hed/config/default_config_910.yaml"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 2 ]
then
    echo "Usage: sh run_single_eval.sh [DEVICE_ID] [CONFIG_PATH]"
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export DEVICE_ID=$1
CONFIG_PATH=$2

rm -rf evalLOG$1
mkdir ./evalLOG$1
cp ./*.py ./evalLOG$1
cp -r ./src ./evalLOG$1
cd ./evalLOG$1 || exit
echo "start eval for device $1"
env > env.log
python eval.py  \
--device_id=$DEVICE_ID \
--config_path=$CONFIG_PATH \
--alg "HED" \
--model_name_list "hed" \
--result_dir result/hed_result \
--save_dir result/hed_eval_result \
--gt_dir /disk3/dataset/BSR/BSDS500/data/groundTruth/test \
--key result \
--file_format .mat \
--workers -1 > log.txt 2>&1 &
cd ../

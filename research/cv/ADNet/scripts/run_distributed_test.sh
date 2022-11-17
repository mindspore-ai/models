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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh RANK_TABLE_FILE RANK_SIZE RANK_START /path/weight_file /path/OTB"
echo "For example: bash run_distributed_test.sh  /path/rank_table.json 16 0 weight_file /data/OTB"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
execute_path=$(pwd)
echo ${execute_path}
self_path=$(cd "$(dirname "$0")" || exit; pwd)
echo ${self_path}

export RANK_TABLE_FILE=$1
export RANK_SIZE=$2
DEVICE_START=$3
WEIGHT_FILE=$4

for((i=0;i<$RANK_SIZE;i++));
do
  export RANK_ID=$i
  export DEVICE_ID=$((DEVICE_START + i))
  echo "Start test for rank $RANK_ID, device $DEVICE_ID."
  if [ -d ${execute_path}/eval_device${DEVICE_ID} ]; then
      rm -rf ${execute_path}/eval_device${DEVICE_ID}
    fi
  mkdir ${execute_path}/eval_device${DEVICE_ID}
  cp -f eval.py ${execute_path}/eval_device${DEVICE_ID}
  cp -rf src ${execute_path}/eval_device${DEVICE_ID}
  cd ${execute_path}/eval_device${DEVICE_ID} || exit
  python3.7 -u eval.py --distributed 'True' --weight_file ${WEIGHT_FILE} --dataset_path $5> eval_log$i 2>&1 &
  cd ..
done
wait
filename=`echo ${WEIGHT_FILE##*/} |awk -F. '{print $1}'`
bboxes_folder="results_on_test_images_part2/${filename}.-0.5"
python3 create_plots.py --bboxes_folder ${execute_path}/${bboxes_folder} > eval_result.txt

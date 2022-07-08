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
if [ $# != 4 ]; then
  echo "Usage: bash run_online_train_gpu.sh [DEVICE_NUM] [DATA_PATH] [SEQ_TXT] [PARENT_CKPT_PATH]"
  exit 1
fi

device_ids=(0 1 2 3 4 5 6 7)
filename=$3
device_num=$1
echo $filename
i=0

cat $filename | while read line
do
  device_id=`expr $i % $device_num`
  let i+=1
  python3 train.py \
  --stage 2 \
  --device_id  ${device_ids[device_id]}\
  --seq_name $line \
  --data_path  $2 \
  --parent_ckpt_path $4 > $line.log 2>&1 &
  echo "train online for seq $line on device ${device_ids[device_id]}"
done

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
if [ $# != 5 ]; then
  echo "Usage: bash run_eval_gpu.sh [DEVICE_ID] [SEQ_TXT] [DATA_PATH] [CKPT_PATH] [PREDICTION_PATH]"
  exit 1
fi

device_id=$1
filename=$2
ckpt_path=`dirname $4`/`basename $4`/
echo $filename
echo $ckpt_path

cat $filename | while read line
do
  seq_ckpt_path=$ckpt_path$line'/checkpoint_online-10000_1.ckpt'
  python3 eval.py \
  --device_id  $device_id \
  --seq_name $line \
  --data_path $3 \
  --ckpt_path $seq_ckpt_path >> eval.log 2>&1 &
  echo "eval for seq $line on device $device_id"
  sleep 20s
done
python3 evaluation_davis.py \
--eval_txt $filename \
--prediction_path $5 \
--gt_path $3  > eval_davis.log 2>&1 &
echo "eval davis"

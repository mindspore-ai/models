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

DIR="$(cd "$(dirname "$0")" && pwd)"

# help message
if [ $# != 7 ]; then
  echo "Usage: bash ./scripts/run_distributed_single_eval.sh [ckpt_start_id] [mode] [dataset] [annotation_path] [video_path] [output_ckpt_path] [train_steps]"
  exit 1
fi

rm -rf $DIR/output_eval
mkdir $DIR/output_eval

for ((i = 0; i <= 8 - 1; i++)); do
  echo 'start eval, device id='${i}'...'
  nohup python ./eval.py \
    --device_id=${i} \
    --test_mode=$2 \
    --$2_path=$6/i3d-$((${i} + $1))_$7.ckpt \
    --dataset=$3 \
    --video_path=$5 \
    --annotation_path=$4 >$DIR/output_eval/eval_divice${i}_log.txt 2>&1 &
done

wait

for ((i = 0; i <= 8 - 1; i++)); do
  cat $DIR/output_eval/eval_divice${i}_log.txt | grep "checkpoint:" >>output_eval/summary.txt
  cat $DIR/output_eval/eval_divice${i}_log.txt | grep "$2 accuracy" >>output_eval/summary.txt
done

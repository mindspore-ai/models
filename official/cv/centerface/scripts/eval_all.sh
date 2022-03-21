#!/bin/sh
# Copyright 2020 Huawei Technologies Co., Ltd
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

if [ $# != 1 ] && [ $# != 5 ]
then
  echo "Usage: bash eval_all.sh [GROUND_TRUTH_PATH]"
  echo "       bash eval_all.sh [GROUND_TRUTH_PATH] [FILTER_EASY](optional) [FILTER_MEDIUM](optional) [FILTER_HARD](optional) [FILTER_SUM](optional)"
  exit 1
fi

FILTER_EASY=0
FILTER_MEDIUM=0
FILTER_HARD=0
FILTER_SUM=0

if [ $# = 5 ]
then
  FILTER_EASY=$2
  FILTER_MEDIUM=$3
  FILTER_HARD=$4
  FILTER_SUM=$5
fi

root=$PWD
save_path=$root/output/centerface/
if [ ! -d $save_path ]
then
    echo "error: save_path=$save_path is not a dir"
exit 1
fi

ground_truth_path=$1
if [ ! -d $ground_truth_path ]
then
    echo "error: ground_truth_path=$ground_truth_path is not a dir"
exit 1
fi

if [ -f log_eval_all.txt ]
then
    rm -rf ./log_eval_all.txt
fi

start_epoch=`ls $save_path | sort -n | head -n 1`
end_epoch=`ls $save_path | sort -n | tail -n 1`

for i in $(seq $start_epoch $end_epoch)
do
    python ../dependency/evaluate/eval.py --pred=$save_path$i --gt=$ground_truth_path >> log_eval_all.txt 2>&1 &
    sleep 10
done
wait

python ../src/find_best_checkpoint.py --result_file_path=./log_eval_all.txt \
--filter_easy=$FILTER_EASY \
--filter_medium=$FILTER_MEDIUM \
--filter_hard=$FILTER_HARD \
--filter_sum=$FILTER_SUM >> log_eval_all.txt 2>&1 &

#! /bin/bash
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

if [ $# != 4 ]
then
  echo "Usage: bash scripts/run_distribute_train.sh /path/to/cityscapes DEVICE_ID RANK_TABLE_FILE"
  echo "Example: bash scripts/run_distribute_train.sh /home/name/cityscapes 4 0,1,2,3 /home/name/rank_table_4pcs.json"
  exit 1
fi

if [ ! -d $1 ]
then
    echo "error: DATASET_PATH=$1 is not a directory"
exit 1
fi

if [ ! -f $4 ]
then
    echo "error: RANK_TABLE_FILE=$4 is not a file"
exit 1
fi

echo "CityScapes dataset path: $1"
echo "RANK_SIZE: $2"
echo "DEVICE_ID: $3"
echo "RANK_TABLE_FILE: $4"

# ps -aux | grep "python -u ../../train.py" | awk '{print $2}' | xargs kill -9

export HCCL_CONNECT_TIMEOUT=600
export RANK_SIZE=$2
cityscapes_path=$1
IFS="," read -r -a devices <<< "$3";
export RANK_TABLE_FILE=$4
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../

if [ -d "distribute_train" ]
then
  echo "delete old logs!"
  rm -rf distribute_train
fi

mkdir distribute_train
cd distribute_train

# 1.train
for((i=0;i<RANK_SIZE;i++))
do
{
  mkdir ./log$i
  cd ./log$i
  export RANK_ID=$i
  export DEVICE_ID=${devices[i]}
  echo "start training for rank $i, device $DEVICE_ID"

  python -u ../../train.py \
    --lr 1e-3 \
    --repeat 2 \
    --run_distribute true \
    --save_path './' \
    --mindrecord_train_data "../../data/train.mindrecord" \
    --stage 1 \
    --ckpt_path "" \
    > log.txt 2>&1
  cd ../
} &
done
wait

# 2.train
for((i=0;i<RANK_SIZE;i++))
do
{
  mkdir ./log$i
  cd ./log$i
  export RANK_ID=$i
  export DEVICE_ID=${devices[i]}
  echo "start training for rank $i, device $DEVICE_ID"

  python -u ../../train.py \
    --lr 1e-3 \
    --repeat 2 \
    --run_distribute true \
    --save_path './' \
    --mindrecord_train_data "../../data/train.mindrecord" \
    --stage 2 \
    --ckpt_path "../log0/Encoder-65_496.ckpt" \
    > log.txt 2>&1
  cd ../
} &
done
wait

# 3.train
for((i=0;i<RANK_SIZE;i++))
do
{
  mkdir ./log$i
  cd ./log$i
  export RANK_ID=$i
  export DEVICE_ID=${devices[i]}
  echo "start training for rank $i, device $DEVICE_ID"

  python -u ../../train.py \
    --lr 1e-3 \
    --repeat 2 \
    --run_distribute true \
    --save_path './' \
    --mindrecord_train_data "../../data/train.mindrecord" \
    --stage 3 \
    --ckpt_path "../log0/Encoder_1-85_496.ckpt" \
    > log.txt 2>&1
  cd ../
} &
done
wait

# 4.train
for((i=0;i<RANK_SIZE;i++))
do
{
  mkdir ./log$i
  cd ./log$i
  export RANK_ID=$i
  export DEVICE_ID=${devices[i]}
  echo "start training for rank $i, device $DEVICE_ID"

  python -u ../../train.py \
    --lr 1e-3 \
    --repeat 2 \
    --run_distribute true \
    --save_path './' \
    --mindrecord_train_data "../../data/train.mindrecord" \
    --stage 4 \
    --ckpt_path "../log0/ERFNet-65_496.ckpt" \
    > log.txt 2>&1
  cd ../
} &
done
wait

# eval
cd ./log0
for((i=0;i<RANK_SIZE;i++))
do
{
  export RANK_ID=$i
  export DEVICE_ID=${devices[i]}
  echo "start eval for rank $i, device $DEVICE_ID"
  python -u ../../eval.py \
    --data_path ${cityscapes_path} \
    --run_distribute true \
    --encode false \
    --model_root_path './' \
    --device_id ${devices[i]} \
    > log${i}_eval.txt 2>&1 &
}
done

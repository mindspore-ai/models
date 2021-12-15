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
echo "bash run_distributed_classifier_ascend.sh"
echo "for example: bash scripts/run_distributed_classifier_ascend.sh"
echo "assessment_method include: [MCC, Spearman_correlation ,Accuracy]"
echo "TASK_TYPE include: [sst-2, mnli]"
echo "NETWORK include: [base, large, xlarge, xxlarge]"
echo "=============================================================================================================="

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
echo $PATH1

if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi

export RANK_TABLE_FILE=$PATH1
export RANK_SIZE=8
export HCCL_CONNECT_TIMEOUT=600

START_DEVICE_NUM=0
TASK_TYPE='sst-2'
NETWORK='base'

mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
ROOT_PATH=${GLOG_log_dir}/${TASK_TYPE}/${TASK_TYPE}_ckpt_eval_epoch_${NETWORK}/

for((i=0; i<$RANK_SIZE; i++))
do
    export DEVICE_ID=`expr $i + $START_DEVICE_NUM`
    export RANK_ID=$i
    echo "start training for rank $i, device $DEVICE_ID"
    python ${PROJECT_DIR}/../run_classifier.py \
    --config_path="../../task_classifier_config.yaml" \
    --device_target="Ascend" \
    --distribute="true" \
    --do_train="true" \
    --do_eval="true" \
    --assessment_method="Accuracy" \
    --device_num=$RANK_SIZE \
    --device_id=$DEVICE_ID \
    --epoch_num=30 \
    --num_class=2 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --save_finetune_checkpoint_path="${ROOT_PATH}" \
    --load_pretrain_checkpoint_path="" \
    --load_finetune_checkpoint_path="${ROOT_PATH}" \
    --train_data_file_path="train.mindrecord" \
    --eval_data_file_path="dev.mindrecord" \
    --schema_file_path="" > ${ROOT_PATH}${TASK_TYPE}_log_${DEVICE_ID}.txt 2>&1 &
done


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
echo "bash scripts/run_classifier.sh"
echo "for example: bash scripts/run_classifier.sh"
echo "assessment_method include: [MCC, Spearman_correlation ,Accuracy]"
echo "=============================================================================================================="

export RANK_SIZE=1

mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

python ${PROJECT_DIR}/../run_classifier.py  \
    --config_path="../../task_classifier_config.yaml" \
    --device_target="Ascend" \
    --do_train="true" \
    --do_eval="true" \
    --assessment_method="Accuracy" \
    --device_id=2 \
    --epoch_num=30 \
    --num_class=2 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --save_finetune_checkpoint_path="" \
    --load_pretrain_checkpoint_path="" \
    --load_finetune_checkpoint_path="" \
    --train_data_file_path="train.mindrecord" \
    --eval_data_file_path="dev.mindrecord" \
    --schema_file_path="" > classfifier_log.txt 2>&1 &

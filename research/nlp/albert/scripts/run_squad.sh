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
echo "bash scripts/run_squad.sh"
echo "for example: bash scripts/run_squad.sh"
echo "assessment_method include: [f1, Exact match]"
echo "=============================================================================================================="

mkdir -p ms_log_squard
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log_squard
export GLOG_logtostderr=0
python ${PROJECT_DIR}/../run_squad_v1.py  \
  --config_path="../../task_squad_config.yaml" \
  --device_target="Ascend" \
  --do_train="false" \
  --do_eval="true" \
  --device_id=1 \
  --epoch_num=10 \
  --num_class=2 \
  --train_data_shuffle="true" \
  --eval_data_shuffle="false" \
  --train_batch_size=48 \
  --eval_batch_size=1 \
  --vocab_file_path="30k-clean.vocab" \
  --save_finetune_checkpoint_path="" \
  --load_pretrain_checkpoint_path="" \
  --load_finetune_checkpoint_path="" \
  --train_data_file_path="train_feature_file_v1.mindrecord" \
  --eval_json_path="dev-v1.1.json" \
  --spm_model_file="30k-clean.model" \
  --predict_feature_left_file="predict_feature_left_file_v1.pkl" \
  --schema_file_path="" > squad_log.txt 2>&1 &

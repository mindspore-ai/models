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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_standalone_td_onnx.sh [config_path]"
echo "for example: bash scripts/run_standalone_td_onnx.sh /home/data1/td_config_sst2.yaml"
echo "=============================================================================================================="

if [ $# != 1 ]; then
  echo "bash scripts/run_standalone_td_onnx.sh [config_path]"
  exit 1
fi
mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python run_eval_onnx.py \
    --config_path=$1 \
    --device_target="GPU" \
    --device_id=0 \
    --do_train="false" \
    --do_eval="true" \
    --td_phase1_epoch_size=10 \
    --td_phase2_epoch_size=3 \
    --do_shuffle="true" \
    --enable_data_sink="true" \
    --data_sink_steps=-1 \
    --max_ckpt_num=1 \
    --load_teacher_ckpt_path="" \
    --load_gd_ckpt_path="" \
    --onnx_path="/home/username/models/research/nlp/tinybert/MNLI.onnx" \
    --train_data_dir="" \
    --eval_data_dir="/home/username/GLUE-baselines/glue_data/MNLI/" \
    --schema_dir="" \
    --dataset_type="tfrecord" \
    --task_type="classification" \
    --task_name="MNLI" \
    --assessment_method="accuracy" > log.txt 2>&1 &

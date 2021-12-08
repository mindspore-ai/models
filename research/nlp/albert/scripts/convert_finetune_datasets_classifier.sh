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

TASK_NAME='WNLI'
SHARD_NUM=1
CUR_DIR=`pwd`

python ${CUR_DIR}/src/classifier_utils.py  \
    --task_name=$TASK_NAME \
    --vocab_path="/albert_base/30k-clean.vocab" \
    --spm_model_file="/albert_base/30k-clean.model" \
    --max_seq_length=512 \
    --do_lower_case="true" \
    --input_dir="" \
    --output_dir="" \
    --shard_num=$SHARD_NUM \
    --do_train="true" \
    --do_eval="true" \
    --do_pred="true" \

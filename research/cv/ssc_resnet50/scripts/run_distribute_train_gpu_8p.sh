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
usage() {
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_distribute_train_ascend_8p.sh ANNOTATION EXP_DIR PRE_TRAINED"
  echo "for example: bash run_distribute_train_ascend_8p.sh /path/to/annotation.json /path/to/save/folder /path/to/pre_trained.ckpt(option) "
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
}


if [ $# -lt 2 ]; then
  usage
  exit 1
fi


ANNOTATION=$1
EXP_DIR=$2
PRE_TRAINED=$3


BASE_MODEL_OUTPUT=$EXP_DIR/base_model
FINAL_MODEL_OUTPUT=$EXP_DIR/final_model
mkdir -p $BASE_MODEL_OUTPUT
mkdir -p $FINAL_MODEL_OUTPUT


# train in 10% data
echo '--------------step 1: start trainning base model-----------------'
bash run_distribute_train_model_gpu.sh 8 $BASE_MODEL_OUTPUT $PRE_TRAINED $ANNOTATION


# sorted data value by trained model
echo '--------------step2: start sorting data in a json-----------------'
BASE_MODEL="$BASE_MODEL_OUTPUT/model_last.ckpt"
bash select_sample_gpu.sh 8 $EXP_DIR $ANNOTATION $BASE_MODEL


# merge data
echo '--------------step3: merge data in a new annotation json-----------------'
python3 ../merge_final_anno.py \
        --class_to_id ../class_to_idx.json \
        --txt_root_path $EXP_DIR \
        --base_json $ANNOTATION


# train final model
echo '--------------step 4: start trainning final model-----------------'
NEW_ANNOTATION="$EXP_DIR/annotation_new.json"
bash run_distribute_train_model_gpu.sh 8 $FINAL_MODEL_OUTPUT $NEW_ANNOTATION $PRE_TRAINED

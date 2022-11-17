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
if [ $# -ne 1 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_eval_onnx.sh [TASK_TYPE]"
    echo "for example: bash scripts/run_eval_onnx.sh chnsenticorp"
    echo "TASK_TYPE including [chnsenticorp, xnli, dbqa]"
    echo "=============================================================================================================="
exit 1
fi

TASK_TYPE=$1
DEVICE_ID=0
CUR_DIR=`pwd`
mkdir -p ms_log
ONNX_PATH=${CUR_DIR}/ernie_finetune.onnx
DATA_PATH=${CUR_DIR}/data
GLOG_log_dir=${CUR_DIR}/ms_log

case $TASK_TYPE in
  "chnsenticorp")
    PY_NAME=run_eval_onnx
    NUM_LABELS=3
    EVAL_BATCH_SIZE=1
    EVAL_DATA_PATH="${DATA_PATH}/chnsenticorp/chnsenticorp_test.mindrecord"
    ASSESSMENT_METHOD="accuracy"
    ;;

  "xnli")
    PY_NAME=run_eval_onnx
    NUM_LABELS=3
    EVAL_BATCH_SIZE=1
    EVAL_DATA_PATH="${DATA_PATH}/xnli/xnli_test.mindrecord"
    ASSESSMENT_METHOD="accuracy"
    ;;

  "dbqa")
    PY_NAME=run_eval_onnx
    NUM_LABELS=2
    EVAL_BATCH_SIZE=1
    EVAL_DATA_PATH="${DATA_PATH}/nlpcc-dbqa/dbqa_test.mindrecord"
    ASSESSMENT_METHOD="f1"
    ;;
esac

python ${CUR_DIR}/${PY_NAME}.py \
    --task_type=${TASK_TYPE} \
    --device_target="GPU" \
    --device_id=${DEVICE_ID} \
    --number_labels=${NUM_LABELS} \
    --eval_data_shuffle="false" \
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    --eval_data_file_path=${EVAL_DATA_PATH} \
    --onnx_file=${ONNX_PATH} \
    --assessment_method=${ASSESSMENT_METHOD} > ${GLOG_log_dir}/${TASK_TYPE}_onnx_log.txt 2>&1 &

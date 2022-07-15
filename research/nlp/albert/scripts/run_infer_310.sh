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

if [[ $# -lt 4 || $# -gt 8 ]]; then
    echo "Usage: bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATA_FILE_PATH] [NEED_PREPROCESS] [TASK_TYPE] [EVAL_JSON_PATH] [VOCAB_FILE_PATH] [SPM_MODEL_FILE] [DEVICE_ID]
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
    EVAL_JSON_PATH is original eval dataset of task squadv1, must be provided if task type is squadv1.
    VOCAB_FILE_PATH is vocabulary file of TensorFlow Albert, must be provided if task type is squadv1.
    SPM MODEL FILE is spm file of TensorFlow Albert, must be provided if task type is squadv1.
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero."
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
eval_data_file_path=$(get_real_path $2)

if [ "$3" == "y" ] || [ "$3" == "n" ];then
    need_preprocess=$3
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

task_type=$4
device_id=0
eval_json_path=None
vocab_file_path=None
spm_model_file=None
if [ $task_type == "squadv1" ];then
    eval_json_path=$(get_real_path $5)
    vocab_file_path=$(get_real_path $6)
    spm_model_file=$(get_real_path $7)
    if [ $# == 8 ]; then
      device_id=$8
    fi
else
  if [ $# == 5 ]; then
    device_id=$5
  fi

fi

echo "mindir name: "$model
echo "eval_data_file_path: "$eval_data_file_path
echo "need preprocess: "$need_preprocess
echo "task type: "$task_type
echo "eval json path: "$eval_json_path
echo "vocab file path: "$vocab_file_path
echo "spm model file: "$spm_model_file
echo "device id: "$device_id

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

function preprocess_data()
{
    if [ -d preprocess_result ]; then
        rm -rf ./preprocess_result
    fi
    mkdir preprocess_result
    python ../preprocess.py --eval_data_file_path=$eval_data_file_path --task_type=$task_type --result_path=./preprocess_result/ --eval_json_path=$eval_json_path --vocab_file_path=$vocab_file_path --spm_model_file=$spm_model_file
}

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_files ]; then
        rm -rf ./result_files
    fi
    if [ -d time_result ]; then
        rm -rf ./time_result
    fi
    mkdir result_files
    mkdir time_result

    ../ascend310_infer/out/main --mindir_path=$model --input0_path=./preprocess_result/00_data --input1_path=./preprocess_result/01_data \
    --input2_path=./preprocess_result/02_data --input3_path=./preprocess_result/03_data --device_id=$device_id &> infer.log

}

function cal_acc()
{
    python ../postprocess.py --result_dir=./result_files --task_type=$task_type \
    --label_dir=./preprocess_result/03_data --eval_data_file_path=$eval_data_file_path --eval_json_path=$eval_json_path \
    --input1_path=./preprocess_result/01_data &> acc.log
}

if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ $? -ne 0 ]; then
        echo "preprocess dataset failed"
        exit 1
    fi
fi
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
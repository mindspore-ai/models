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

if [[ $# -lt 7 || $# -gt 11 ]]; then
    echo "Usage: bash run_infer_cpp.sh [MINDIR_PATH] [LABEL_PATH] [DATA_FILE_PATH] [DATASET_FORMAT] [SCHEMA_PATH] [TASK]
    [VOCAB_FILE_PATH] [EVAL_JSON_PATH] [NEED_PREPROCESS] [DEVICE_TYPE] [DEVICE_ID]
    TASK is mandatory, and must choose from [ner|ner_crf|classifier|squad]
    When TASK is squad, [VOCAB_FILE_PATH] and [EVAL_JSON_PATH] is needed.
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
    DEVICE_TYPE can choose from [Ascend, GPU, CPU]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ -z "$1" ]; then
        echo ""
    elif [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
label_file_path=$(get_real_path $2)
eval_data_file_path=$(get_real_path $3)
dataset_format=$4
schema_file_path=$(get_real_path $5)
net_type=${6,,}
if [ $net_type == 'ner' ]; then
  echo "downstream: NER"
elif [ $net_type == 'ner_crf' ]; then
  echo "downstream: NER-CRF"
elif [ $net_type == 'classifier' ]; then
  echo "downstream: Classifier"
elif [ $net_type == 'squad' ]; then
  echo "downstream: SQuAD"
  vocab_file_path=$7
  eval_json_path=$8
  echo "vocab_file_path "$vocab_file_path
  echo "eval_json_path: "$eval_json_path
else
  echo "[TASK] must choose from [ner|ner_crf|classifier|squad]"
  exit 1
fi

if [ "$7" == "y" ] || [ "$7" == "n" ];then
    need_preprocess=$7
elif [ "$9" == "y" ] || [ "$9" == "n" ];then
    need_preprocess=$9
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

device_id=0
if [ $# == 9 ]; then
    device_id=$9
fi
if [ $# == 11 ]; then
    device_id=${11}
fi

if [ $net_type == 'squad' ]; then
    if [ ${10} == 'Ascend' ] || [ ${10} == 'GPU' ] || [ ${10} == 'CPU' ]; then
    device_type=${10}
    else
    echo "DEVICE_TYPE can choose from [Ascend, GPU, CPU]"
    exit 1
    fi

    if [ ${10} == 'GPU' ]; then
        device_id=0
    fi
else
    if [ $8 == 'Ascend' ] || [ $8 == 'GPU' ] || [ $8 == 'CPU' ]; then
    device_type=$8
    else
    echo "DEVICE_TYPE can choose from [Ascend, GPU, CPU]"
    exit 1
    fi

    if [ $8 == 'GPU' ]; then
        device_id=0
    fi
fi
echo "device type: "$device_type

if [ $MS_LITE_HOME ]; then
  RUNTIME_HOME=$MS_LITE_HOME/runtime
  TOOLS_HOME=$MS_LITE_HOME/tools
  RUNTIME_LIBS=$RUNTIME_HOME/lib:$RUNTIME_HOME/third_party/glog/:$RUNTIME_HOME/third_party/libjpeg-turbo/lib
  RUNTIME_LIBS=$RUNTIME_LIBS:$RUNTIME_HOME/third_party/dnnl/
  export LD_LIBRARY_PATH=$RUNTIME_LIBS:$TOOLS_HOME/converter/lib:$LD_LIBRARY_PATH
  echo "Insert LD_LIBRARY_PATH the MindSpore Lite runtime libs path: $RUNTIME_LIBS $TOOLS_HOME/converter/lib"
fi

echo "mindir name: "$model
echo "label_file_path: "$label_file_path
echo "eval_data_file_path: "$eval_data_file_path
echo "dataset_format: "$dataset_format
echo "schema_file_path: "$schema_file_path
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --task=$net_type --do_eval=true --label_file_path=$label_file_path --eval_data_file_path=$eval_data_file_path \
        --dataset_format=$dataset_format --schema_file_path=$schema_file_path --result_path=./preprocess_Result/
}

function compile_app()
{
    cd ../cpp_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result

    ../cpp_infer/out/main --device_type=$device_type --mindir_path=$model --input0_path=./preprocess_Result/00_data --input1_path=./preprocess_Result/01_data \
        --input2_path=./preprocess_Result/02_data --input3_path=./preprocess_Result/03_data --task=$net_type --device_id=$device_id &> infer.log

}

function cal_acc()
{
    if [ $net_type == 'classifier' ]; then
        python ../postprocess.py --result_path=./result_Files --label_dir=./preprocess_Result/03_data --task=$net_type \
            --seq_length=1 --assessment_method=Accuracy &> acc.log
    elif [ $net_type == 'squad' ]; then
        python ../postprocess.py --result_path=./result_Files --preprocess_path=./preprocess_Result --task=$net_type \
            --seq_length=384 --vocab_file_path=$vocab_file_path --eval_json_path=$eval_json_path &> acc.log
    else
        python ../postprocess.py --result_path=./result_Files --label_dir=./preprocess_Result/03_data --task=$net_type &> acc.log
    fi
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
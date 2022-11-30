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
echo "bash run_standalone_eval_onnx.sh TEST_DATASET ONNX_FILE \
  VOCAB_ADDR BPE_CODE_ADDR TEST_TARGET"
echo "for example:"
echo "bash run_standalone_eval_onnx.sh \
  /home/workspace/dataset_menu/newstest2014.en.mindrecord \
  /home/workspace/seq2seq/seq2seq.onnx \
  /home/workspace/wmt14_fr_en/vocab.bpe.32000 \
  /home/workspace/wmt14_fr_en/bpe.32000 \
  /home/workspace/wmt14_fr_en/newstest2014.fr"
echo "It is better to use absolute path."
echo "=============================================================================================================="

if [ $# != 5 ]; then
 echo "bash run_standalone_eval_onnx.sh \
  /home/workspace/dataset_menu/newstest2014.en.mindrecord \
  /home/workspace/seq2seq/seq2seq.onnx \
  /home/workspace/wmt14_fr_en/vocab.bpe.32000 \
  /home/workspace/wmt14_fr_en/bpe.32000 \
  /home/workspace/wmt14_fr_en/newstest2014.fr"
 exit 1
fi

TEST_DATASET=$1
ONNX_FILE=$2
VOCAB_ADDR=$3
BPE_CODE_ADDR=$4
TEST_TARGET=$5

current_exec_path=$(pwd)
echo ${current_exec_path}


export GLOG_v=2

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp -r ../src ./eval
cp -r ../config ./eval
cd ./eval || exit
echo "start for evaluation"
env > env.log
python3 eval_onnx.py \
  --config=${current_exec_path}/eval/config/config_test.json \
  --test_dataset=$TEST_DATASET \
  --onnx_file=$ONNX_FILE \
  --vocab=$VOCAB_ADDR \
  --bpe_codes=$BPE_CODE_ADDR \
  --test_tgt=$TEST_TARGET >log_infer.log 2>&1 &
cd ..

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


if [ $# != 6 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_eval.sh [CONTENT_PATH] [STYLE_PATH] [DECODER_PATH] [EMBEDDING_PATH] [TRANS_PATH] [OUTPUT_PATH]"
    echo "for example: bash scripts/run_eval.sh ./dataset/COCO2014/val2014 ./dataset/wikiart/test save_model/decoder_160000.ckpt save_model/embedding_160000.ckpt save_model/transformer_160000.ckpt ./output"
    echo "=============================================================================================================="
exit 1
fi


CONTENT_PATH=$1
STYLE_PATH=$2
DECODER_PATH=$3
EMBEDDING_PATH=$4
TRANS_PATH=$5
OUTPUT_PATH=$6


python eval.py  \
  --content_dir=$CONTENT_PATH \
  --style_dir=$STYLE_PATH\
  --decoder_path=$DECODER_PATH\
  --trans_path=$TRANS_PATH\
  --embedding_path=$EMBEDDING_PATH\
  --output=$OUTPUT_PATH\
  > eval.log 2>&1 &

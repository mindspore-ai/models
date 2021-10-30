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

"""
convert dataset
"""
if [ $# -ne 1 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash convert_dataset.sh [TASK_NAME]"
    echo "for example: sh scripts/convert_dataset.sh match_kn_gene"
    echo "TASK_TYPE including [match, match_kn, match_kn_gene]"
    echo "=============================================================================================================="
exit 1
fi
TASK_NAME=$1

case $TASK_NAME in
    "match")
        DICT_NAME="data/char.dict"
        ;;
    "match_kn")
        DICT_NAME="data/char.dict"
        ;;
    "match_kn_gene")
        DICT_NAME="data/gene.dict"
        ;;
    esac

python src/reader.py --task_name=${TASK_NAME} \
                     --max_seq_len=256 \
                     --vocab_path=${DICT_NAME} \
                     --input_file=data/build.train.txt \
                     --output_file=data/train.mindrecord
python src/reader.py --task_name=${TASK_NAME} \
                     --max_seq_len=256 \
                     --vocab_path=${DICT_NAME} \
                     --input_file=data/build.dev.txt \
                     --output_file=data/dev.mindrecord
python src/reader.py --task_name=${TASK_NAME} \
                     --max_seq_len=256 \
                     --vocab_path=${DICT_NAME} \
                     --input_file=data/build.test.txt \
                     --output_file=data/test.mindrecord


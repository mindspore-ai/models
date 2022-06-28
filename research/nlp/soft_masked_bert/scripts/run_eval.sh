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

DIR="$(cd "$(dirname "$0")" && pwd)"

# help message
if [ $# != 2 ]; then
  echo "Usage: bash scripts/run_eval.sh [bert_ckpt] [ckpt_dir]"
  exit 1
fi

rm -rf $DIR/output_eval
mkdir $DIR/output_eval

export BERT_CKPT=$1
export CKPT_DIR=$2

nohup python eval.py --bert_ckpt ${BERT_CKPT} --ckpt_dir ${CKPT_DIR}  >$DIR/output_eval/eval_log.txt 2>&1 &

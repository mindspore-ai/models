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

if [ $# -lt 3 ]
then
  echo "Usage: bash run_eval_gpu.sh [EVALDATA_PATH] [CKPT_DIR] [DEVICE_ID]"
  exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$3

cd ..

python ${BASEPATH}/../eval.py \
          --eval_dir=$1 \
          --device_target='GPU' \
          --ckpt_dir=$2 > eval.log  2>&1 &

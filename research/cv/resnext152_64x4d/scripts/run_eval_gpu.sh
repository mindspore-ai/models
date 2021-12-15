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

if [ $# != 2 ]
then
    echo "===================================================================================================="
    echo "Please run the script as: "
    echo "bash run_eval_gpu.sh DATA_DIR PATH_CHECKPOINT"
    echo "for example: bash run_eval_gpu.sh /path/ImageNet2012/val /path/trained/model.ckpt"
    echo "===================================================================================================="
    exit 1
fi

DATA_DIR=$1
PATH_CHECKPOINT=$2

python eval.py  \
    --pretrained=$PATH_CHECKPOINT \
    --platform=GPU \
    --data_dir=$DATA_DIR > eval_log.txt 2>&1 &

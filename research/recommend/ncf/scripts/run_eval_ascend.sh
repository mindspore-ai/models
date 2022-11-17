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
if [ $# != 3 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_eval_ascend.sh DATASET_PATH CKPT_FILE DEVICE_ID"
    echo "for example: bash scripts/run_eval_ascend.sh /dataset_path NCF-25_19418.ckpt 0"
exit 1
fi

data_path=$1
ckpt_file=$2
export DEVICE_ID=$3
python ./eval.py --data_path $data_path --dataset 'ml-1m'  --eval_batch_size 160000 \
    --output_path './output/' --eval_file_name 'eval.log' --checkpoint_file_path $ckpt_file \
    --device_target=Ascend --device_id $DEVICE_ID > eval.log 2>&1 &

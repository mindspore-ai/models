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
if [ $# != 3 ];
then
  echo "no enough params"
  exit
fi
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval.sh OUTPUT_PATH DATANAME MODEL_CKPT DEVICE_ID"
echo "for example: bash run_eval.sh /path/output /path/model.ckpt 0"
echo "It is better to use absolute path."
echo "=============================================================================================================="
cd ../
OUTPUT_PATH=$1
MODEL_CKPT=$2
python eval.py --output_path $OUTPUT_PATH  --model_ckpt $MODEL_CKPT> eval_tacotron2.log 2>&1 &

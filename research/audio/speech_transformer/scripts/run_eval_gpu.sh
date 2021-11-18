#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
if [ $# != 4 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval_gpu.sh [DEVICE_ID] [DATA_JSON_PATH] [CKPT_PATH] [CONFIG_PATH]"
echo "for example: bash run_eval_gpu.sh 0 /your/path/data.json /your/path/checkpoint_file ./default_config.yaml"
echo "Note: set the checkpoint and dataset path in default_config.yaml"
echo "=============================================================================================================="
exit 1;
fi

export CONFIG_PATH=$4
DEVICE_ID=$1

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH3=$(get_real_path $2)
PATH4=$(get_real_path $3)
echo $PATH3
echo $PATH4

python eval.py  \
    --config_path=$CONFIG_PATH \
    --device_target="GPU" \
    --device_id=$DEVICE_ID \
    --data_json_path=$PATH3 \
    --model_file=$PATH4 > eval.log 2>&1 &

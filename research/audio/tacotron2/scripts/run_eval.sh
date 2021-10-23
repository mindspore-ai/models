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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_eval.sh OUTPUT_PATH DATANAME MODEL_CKPT DEVICE_ID"
echo "for example: bash run_eval.sh output ljspeech device0/ckpt0/tacotron2-5-118.ckpt 0"
echo "It is better to use absolute path."
echo "=============================================================================================================="

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
OUTPUT_PATH=$1
echo $PWD/eval/$OUTPUT_PATH
DATANAME=$2
MODEL_CKPT=$(get_real_path $3)
DEVICEID=$4
export DEVICE_NUM=1
export DEVICE_ID=$DEVICEID
export RANK_ID=0
export RANK_SIZE=1

config_path="./${DATANAME}_config.yaml"
echo "config path is : ${config_path}"

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir -p ./eval/$OUTPUT_PATH
cp ../*.py ./eval
cp ../*.yaml ./eval
cp -r ../src ./eval
cp -r ../model_utils ./eval
cp -r ../scripts/*.sh ./eval
cd ./eval || exit
echo "start evaling for device $DEVICE_ID"
env > env.log
python ../../eval.py --config_path $config_path --output_path $PWD/$OUTPUT_PATH  --model_ckpt $MODEL_CKPT> eval_tacotron2.log 2>&1 &
cd ..

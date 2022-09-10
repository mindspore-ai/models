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

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:${s}[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  "$1" |
   awk -F"$fs" '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'"$prefix"'",vn, $2, $3);
      }
   }'
}

if [ $# != 2 ]; then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_eval.sh CONFIG_PATH CKPT_PATH"
    echo "For example: run_eval.sh ../config/default_gpu.yaml /output_dir/exp_sgd_optim/ckpt/bmn_auto_-9_603.ckpt"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi

CONFIG_PATH=$(get_real_path "$1")

if [ ! -f "$CONFIG_PATH" ]; then
    echo "error: CKPT_PATH=$CONFIG_PATH is not a file"
    exit 1
fi

CKPT_PATH=$(get_real_path "$2")
experiment_name=""

eval "$(parse_yaml "$CONFIG_PATH")"

EXP_NAME="$(cut -d' ' -f2 <<<"$experiment_name")"

if [ ! -f "$CKPT_PATH" ]; then
    echo "error: CKPT_PATH=$CKPT_PATH is not a file"
    exit 1
fi



EVAL_DIR=$EXP_NAME"_eval"

if [ -d "$EVAL_DIR" ]; then
    rm -rf "$EVAL_DIR"
fi
mkdir ./"$EVAL_DIR"
cp ../*.py ./"$EVAL_DIR"
cp -r ../config ./"$EVAL_DIR"
cp -- *.sh ./"$EVAL_DIR"
cp -r ../src ./"$EVAL_DIR"
cd ./"$EVAL_DIR" || exit
env > env.log

python eval.py \
    --config_path "$CONFIG_PATH" \
    --eval.checkpoint "$CKPT_PATH" &> log &

echo "Done!"

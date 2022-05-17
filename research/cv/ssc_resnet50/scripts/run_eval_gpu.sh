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
usage() {
  echo "=============================================================================================================="
  echo "Please run the script as: "
  echo "bash run_eval_gpu.sh DEVICE_ID CKPT_PATH FOLDER"
  echo "for example: bash run_eval_gpu.sh 0 /path/to/checkpoint /path/to/eval/dataset/dir 0/1"
  echo "It is better to use absolute path."
  echo "=============================================================================================================="
}


if [ $# -lt 4 ]; then
  usage
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

ckpt_path=$(get_real_path $2)

test_root=$3

if [ $4 == 1 ] && [ ! -d $ckpt_path ]; then
  echo -e "error: CKPT_PATH=$ckpt_path is not a folder \n"
  exit 1
fi

if [ $4 == 0 ] && [ ! -f $ckpt_path ]; then
  echo -e "error: CKPT_PATH=$ckpt_path is not a file \n"
  exit 1
fi


# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=3
export PYTHONUNBUFFERED=1
eval_device_id=$1

current_time=$(date +%Y%m%d-%H%M%S)
prefix_dir="$(pwd)/eval_1p_${current_time}"
ulimit -u unlimited
echo "Process: device$eval_device_id ..."


mkdir -p ${prefix_dir}/device$eval_device_id
mkdir -p ${prefix_dir}/device$eval_device_id/outputs/
cp ../eval.py ${prefix_dir}/device$eval_device_id
cp ../*.yaml ${prefix_dir}/device$eval_device_id
cp -r ../src/ ${prefix_dir}/device$eval_device_id
cd ${prefix_dir}/device$eval_device_id
export DEVICE_ID=$eval_device_id
export CKPT_PATH=$ckpt_path
echo "start eval $ckpt_path for device $eval_device_id"
env >env_log.txt

python eval.py \
       --device_target="GPU" \
       --folder=$4 \
       --eval_pre_trained=$ckpt_path \
       --test_root=$test_root \
       --device_id=${DEVICE_ID} >log.txt 2>&1 &
cd -

echo -e "Process start OK, save in ${prefix_dir} \n"

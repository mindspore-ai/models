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


if [ $# != 2 ]
then 
    echo "Usage: bash run_process_train_data.sh [TRAIN_DATASET_PATH] [OUTPUT_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
if [ ! -d $PATH1 ]
then 
    echo "error: TRAIN_DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

data_path=$1
out_path=$2

out_path=$2
if [ -d $out_path ]; then
    rm -rf $out_path
fi
mkdir $out_path

shopt -s globstar

for i in $data_path/**/*.flac
do
sox $i -r 16000 -c 1 -t sw -
done > $out_path/input.s16

../third_party/out/dump_data -qtrain $out_path/input.s16 $out_path/features.f32 $out_path/data.s16
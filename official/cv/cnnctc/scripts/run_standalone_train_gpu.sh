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

if [ $# != 0 ] && [ $# != 1 ]
then 
    echo "run as bash scripts/run_standalone_train_gpu.sh PRE_TRAINED(options)"
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
echo $PATH1

export DEVICE_NUM=1
export RANK_SIZE=1

ulimit -u unlimited

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ./*.py ./train
cp ./scripts/*.sh ./train
cp -r ./src ./train
cp ./*yaml ./train
cd ./train || exit
env > env.log
if [ -f $PATH1 ]
then
  python train.py --device_target="GPU" --PRED_TRAINED=$PATH1 --run_distribute=False &> log &
else
  python train.py --device_target="GPU" --run_distribute=False &> log &
fi
cd .. || exit

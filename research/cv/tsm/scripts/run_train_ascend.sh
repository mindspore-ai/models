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
ulimit -u unlimited

export DEVICE_NUM=8
export RANK_SIZE=8
export HCCL_CONNECT_TIMEOUT=600
export RANK_TABLE_FILE='/home/wks/hccl_8p_01234567_127.0.0.1.json'
export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

# remove old train_parallel files

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

echo "device num=$DEVICE_NUM"
cd ./scripts
rm -rf train_parallel
mkdir train_parallel
cd ./train_parallel
for((i=0; i<${DEVICE_NUM}; i++))
do
    j=$(($i))
    export DEVICE_ID=${j}  
    # {j}

    export RANK_ID=$((rank_start + i))
    # mkdirs
    mkdir train_parallel$i
    cd ./train_parallel$i
    cp ../../../*.py ./
    cp -r ../../../src ./
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python -u train.py --config_path=../config/tsm_sthv2_config_ascend.yaml > log.txt 2>&1 &
    cd ..
done

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
device_id=$2
data_path=$1
root_path=$(pwd)
log_dir=${root_path}/standalone
if [ ! -d $log_dir ]; then
    mkdir $log_dir
else
    echo $log_dir exist
fi
rm ${root_path}/standalone/device${device_id} -rf
mkdir ${root_path}/standalone/device${device_id}
cd ${root_path}/standalone/device${device_id} || exit
python3 ${root_path}/train.py -dist 'false' -d ${root_path}/${data_path} --device_id ${device_id} -p '' >single.log 2>&1 &

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

air_path=$1
om_path=$2

# Help information. Don't edit it!
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:bash ./ATC_AIR_2_OM.sh <args>"
    echo " "
    echo "parameter explain:
    --model                  set model place, e.g. --model=/home/xj_mindx/lixiang/protonet.air
    --output                 set the name and place of OM model, e.g. --output=/home/HwHiAiUser/fixmatch310_tune4
    --soc_version            set the soc_version, default: --soc_version=Ascend310
    --input_shape            set the input node and shape, default: --input_shape=\"x:1,1,28,28\"
    --insert_op_conf         set the aipp config file, e.g. --insert_op_conf=aipp_opencv.cfg
    -h/--help                show help message
    "
    exit 1
fi



rm -rf ../data/model
mkdir -p ../data/model

echo "Input AIR file path: ${air_path}"
echo "Output OM file path: ${om_path}"
    
atc --input_format=NCHW --framework=1 --model="${air_path}" \
    --input_shape="x:1,1,28,28" --output="${om_path}/protonet" \
    --soc_version=Ascend310 --disable_reuse_memory=1

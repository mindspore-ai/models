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

if [ $# -ne 3 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_AIR_PATH] [OUTPUT_OM_PATH_NAME] [MODALITY]"
  echo "Example: "
  echo "         bash convert_om.sh  xxx.air xx xx"

  exit 1
fi

input_air_path=$1
output_om_path=$2

if [ "$3" == "Flow" ] || [ "$3" == "RGB" ] || [ "$3" == "RGBDiff" ]
then
    if [ "$3" == "Flow" ];then
      input_shape="input:1,10,224,224"
    fi
    if [ "$3" == "RGB" ];then
      input_shape="input:1,3,224,224"
    fi
    if [ "$3" == "RGBDiff" ];then
      input_shape="input:1,18,224,224"
    fi
else
  echo "device_target must be in  ['Flow', 'RGB', 'RGBDiff']"
  exit 1
fi

export install_path=/usr/local/Ascend/

export ASCEND_ATC_PATH=${install_path}/atc
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/latest/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg

atc --framework=1 \
    --input_format=NCHW \
    --model=${input_air_path} \
    --input_shape=${input_shape}  \
    --output=${output_om_path} \
    --log=error \
    --soc_version=Ascend310 \
    --op_select_implmode=high_precision \
    --output_type=FP32

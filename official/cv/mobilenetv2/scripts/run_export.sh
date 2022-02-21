#!/usr/bin/env bash
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


if [ $# -ne 4 ]
then
    echo "Usage:
          bash run_export.sh [PLATFORM] [CHECKPOINT_PATH] [EXPORTED_FORMAT] [EXPORTED_MODEL_PATH]"
exit 1
fi

if [ $1 = "CPU" ] ; then
    CONFIG_FILE="../default_config_cpu.yaml"
elif [ $1 = "GPU" ] ; then
    CONFIG_FILE="../default_config_gpu.yaml"
elif [ $1 = "Ascend" ] ; then
    CONFIG_FILE="../default_config.yaml"
else
    echo "Unsupported platform."
fi;

python ../export.py \
  --config_path=$CONFIG_FILE \
  --run_distribute False \
  --platform=$1 \
  --ckpt_file=$2 \
  --file_format=$3 \
  --file_name=$4

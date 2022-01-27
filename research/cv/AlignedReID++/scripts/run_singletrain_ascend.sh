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
echo "Please run the script at the diractory same with train.py: "
echo "bash ./scripts/run_singletrain_ascend.sh data_url pre_trained"
echo "For example: bash ./scripts/run_singletrain_ascend.sh /path/dataset/ /path/resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44.ckpt"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

data_url=$1
pre_trained=$2
device_id=$3

EXEC_PATH=$(pwd)

python ${EXEC_PATH}/train.py --is_distributed=False --num_instances=4 --data_url=$data_url --pre_trained=$pre_trained --device_id=$device_id > trainsingle.log 2>&1 &
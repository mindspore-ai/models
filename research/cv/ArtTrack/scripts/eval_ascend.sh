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
if [ $# -ne 2 ]
then
    echo "Please run the script as: "
    echo "bash scripts/eval_ascend.sh [mpii_single or coco_multi] [CKPT_PATH]"
    echo "For example: bash scripts/eval_ascend.sh mpii_ single ckpt/rand_0/arttrack-1_356.ckpt"
exit 1
fi
python eval.py "$1" --config config/mpii_eval_ascend.yaml --option "$2" --device_target Ascend

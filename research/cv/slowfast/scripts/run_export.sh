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
if [ $# != 2 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_export.sh CFG CHECKPOINT_FILE_PATH"
echo "for example: bash scripts/run_export.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml slowfast-20_3056.ckpt"
echo "=============================================================================================================="
exit 1;
fi
CFG=$(realpath $1)
CHECKPOINT_FILE_PATH=$(realpath $2)
python -u export.py --cfg "$CFG" \
     TEST.CHECKPOINT_FILE_PATH "$CHECKPOINT_FILE_PATH"

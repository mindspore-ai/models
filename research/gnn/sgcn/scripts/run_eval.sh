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

if [[ $# -gt 3 ]]; then
    echo "Usage: bash ./scripts/run_eval.sh [checkpoint_auc] [checkpoint_f1] [dataset]"
exit 1
fi

if [ ! -d "logs" ]; then
        mkdir logs
fi

nohup python -u eval.py --checkpoint_auc=$1 --checkpoint_f1=$2 --edge_path=$3 --features-path=$3 > logs/eval.log 2>&1 &

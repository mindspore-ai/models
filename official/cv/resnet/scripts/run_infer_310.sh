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

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [NET_TYPE] [DATASET] [DATA_PATH] [CONFIG_PATH] [DEVICE_ID]
    NET_TYPE can choose from [resnet18, resnet34, se-resnet50, resnet50, resnet101, resnet152]
    DATASET can choose from [cifar10, imagenet]
    DEVICE_ID is optional, it can be set by environment variable DEVICE_ID, otherwise the value is zero"
exit 1
fi

shell_dir=$(cd "$(dirname $0)";pwd)
echo "Shell dir " $shell_dir

export DEVICE_TYPE=Ascend

bash $shell_dir/run_infer_cpp.sh "$@"

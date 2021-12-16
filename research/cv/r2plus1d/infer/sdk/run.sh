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

# help message
if [ $# != 2 ]; then
  echo "Usage: bash run.sh [pipeline] [dataset_root_path]"
  exit 1
fi

get_real_path_name() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

DIR="$( cd "$( dirname "$0"  )" && pwd  )"
python3 $DIR/main.py --pipeline="$(get_real_path_name $1)" \
                    --dataset_root_path="$(get_real_path_name $2)"
exit 0

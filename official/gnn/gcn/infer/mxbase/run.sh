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
if [ $# != 1 ]
then
    echo "Usage: sh run.sh [DATASET_NAME]"
exit 1
fi

DATASET_NAME=$1
echo $DATASET_NAME

# run
if [ $DATASET_NAME = cora ]; then
./build/gcn ../data/input/cora/ 1 ../data/model/cora.om 2708 1433 7
else
./build/gcn ../data/input/citeseer/ 1 ../data/model/citeseer.om 3312 3703 6
fi
exit 0
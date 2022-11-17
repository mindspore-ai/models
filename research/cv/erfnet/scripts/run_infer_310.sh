#! /bin/bash
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

if [ $# != 5 ]
then
  echo "Usage: bash scripts/run_infer_310.sh MINDIR IMGS RES LABEL DEVICE_ID"
  echo "Example: bash scripts/run_infer_310.sh /path/to/net.mindir /path/to/images /path/to/result  /path/to/label  0"
  exit 1
fi

if [ ! -f $1 ]
then 
    echo "error: mindir_path=$1 is not a file"
exit 1
fi

if [ ! -d $2 ]
then 
    echo "error: images_path=$2 is not a directory"
exit 1
fi

if [ ! -d $3 ]
then 
    echo "error: result_path=$3 is not a directory"
exit 1
fi

if [ ! -d $4 ]
then 
    echo "error: label_path=$4 is not a directory"
exit 1
fi

echo "model mindir: $1"
echo "images path: $2"
echo "result path: $3"
echo "laebl path:  $4"
echo "device id: $5"

cd ascend310_infer/src
bash build.sh
./build/erfnet $1 $2 $3 $5

cd ../..
python src/eval310.py --res_path $3 --label_path  $4

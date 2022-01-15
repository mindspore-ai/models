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
#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh CONTENT_PATH LABEL_PATH RANK_TABLE_FILE"
echo "For example: bash run_distribute_train.sh /path/to/content /path/to/label"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

if [ ! -d $1 ]
then
    echo "error: CONTENT_PATH=$2 is not a directory"
exit 1
fi

if [ ! -d $2 ]
then
    echo "error: LABEL_PATH=$2 is not a directory"
exit 1
fi

if [ ! -f $3 ]
then
    echo "error: RANK_TABLE_FILE=$3 is not a file"
exit 1
fi

set -e
export RANK_TABLE_FILE=$3
export RANK_SIZE=8
EXEC_PATH=$(pwd)

echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    cd ../
    cp ../*.py ./device$i
    cp -r ../src ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py  --run_distribute 1 --content_path $1 --label_path $2 > output.log 2>&1 &
    echo "$i finish"
    cd ../
done


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
echo "bash run.sh DATA_PATH RANK_SIZE"
echo "For example: bash run.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
export RANK_TABLE_FILE=$1
export RANK_SIZE=$2
EXEC_PATH=$(pwd)

echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf Bookdevice$i
    mkdir Bookdevice$i
    cd ./Bookdevice$i
    mkdir src
    cd ../
    cp ../*.py ./Bookdevice$i
    cp -r ../ckpt ./Bookdevice$i
    cp ../src/*.py ./Bookdevice$i/src
    cp -r ../Books ./Bookdevice$i/Books
    cp -r ../dataset_mindrecord ./Bookdevice$i/dataset_mindrecord
    cd ./Bookdevice$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train_eval.py --mindrecord_path=./dataset_mindrecord --dataset_type=Books --dataset_file_path=./Books --is_modelarts=False --run_distribute=True --train_test=train > output.log 2>&1 &
    echo "$i finish"
    cd ../
done
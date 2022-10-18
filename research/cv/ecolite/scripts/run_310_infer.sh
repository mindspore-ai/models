#!/bin/bash
#copyright 2021 Huawei Technologies Co., Ltd
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
if [ $# != 4 ]
then
    echo "Usage: bash run_310_infer.sh [MINDIR_PATH] [EVAL_DATA_DIR] [DEVICE_ID] [BATCH_SIZE]."
    exit 1
fi

if [ -d $2 ];
then
    rm -rf $2
fi

mkdir $2

if [ -d 'label' ];
then
    rm -rf 'label'
fi

mkdir 'label'

echo "get dataset"
python get_310_eval_dataset.py ucf101 RGB ./data/ucf101_rgb_val_split_1.txt $2
echo "done"


echo "compile..."

cd ./ascend310_infer || exit
bash build.sh &> build.log

echo "compile done"


echo "infer..."

if [ -d result_Files ];
then
    rm -rf result_Files
fi

if [ -d time_Result ];
then
    rm -rf time_Result
fi

mkdir result_Files
mkdir time_Result

dir=$2
if [ ${dir:0:1} = "/" ] 
then 
    ./out/main $1 $2 $3
else 
    ./out/main .$1 .$2 $3
fi 

echo "cal acc..."
resultpath='./result_Files'
labelpath='../label/'
python ../postprocess.py --result_path ${resultpath} --label_path ${labelpath} --batch_size $4 > acc.log 2>&1 &

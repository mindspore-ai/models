#!/bin/bash
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

if [ $# != 1 ]
then
    echo "Usage: bash run_eval.sh [NETWORK] "
exit 1
fi

get_real_path(){
    BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
    path=$(dirname ${BASE_PATH})
    echo $path/
}
path=$(get_real_path)  
export DISTRIBUTE=0
export DEVICE_ID=0
export nnUNet_raw_data_base="$path/src/nnUNetFrame/DATASET/nnUNet_raw"
export nnUNet_preprocessed="$path/src/nnUNetFrame/DATASET/nnUNet_preprocessed"
export RESULTS_FOLDER="$path/src/nnUNetFrame/DATASET/nnUNet_trained_models"

cd $path
python eval.py -i $path/src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task004_Hippocampus/imagesVal/ -o $path/src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task004_Hippocampus/inferTs -t 4 -m $1 -f 0  >> eval.log 2>&1 &


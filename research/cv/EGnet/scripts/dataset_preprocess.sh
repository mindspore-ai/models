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

# The number of parameters transferred is not equal to the required number, print prompt information
if [ $# != 2 ]
then 
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash data_crop.sh [DATA_ROOT] [OUTPUT_ROOT]"
    echo "for example: bash data_crop.sh /data/ ./data_crop/"
    echo "================================================================================================================="
exit 1
fi

# Get absolute path
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

# Get current script path
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
DATA_ROOT=$(get_real_path $1)
OUTPUT_ROOT=$(get_real_path $2)

cd $DATA_ROOT
mkdir tmp
TMP_ROOT=$DATA_ROOT/tmp

# DUT-OMRON
mkdir $TMP_ROOT/DUT-OMRON
cp -r DUT-OMRON-image/DUT-OMRON-image $TMP_ROOT/DUT-OMRON/images
# ground_truth_mask 
cp -r DUT-OMRON-image/pixelwiseGT-new-PNG $TMP_ROOT/DUT-OMRON/ground_truth_mask
# ECSSD nothing

#HKU-IS
mkdir $TMP_ROOT/HKU-IS
cp -r HKU-IS/imgs $TMP_ROOT/HKU-IS/images
cp -r HKU-IS/gt $TMP_ROOT/HKU-IS/ground_truth_mask

#PASCAL-S
mkdir $TMP_ROOT/PASCAL-S
mkdir $TMP_ROOT/PASCAL-S/ground_truth_mask
mkdir $TMP_ROOT/PASCAL-S/images
cp PASCAL-S/Imgs/*.png $TMP_ROOT/PASCAL-S/ground_truth_mask
cp PASCAL-S/Imgs/*.jpg $TMP_ROOT/PASCAL-S/images

# SOD
mkdir $TMP_ROOT/SOD
mkdir $TMP_ROOT/SOD/ground_truth_mask
mkdir $TMP_ROOT/SOD/images
cp SOD/Imgs/*.png $TMP_ROOT/SOD/ground_truth_mask/
cp SOD/Imgs/*.jpg $TMP_ROOT/SOD/images/


cd $BASE_PATH/..
python data_crop.py --data_name=ECSSD  --data_root="$DATA_ROOT" --output_path="$OUTPUT_ROOT"
python data_crop.py --data_name=SOD  --data_root="$TMP_ROOT" --output_path="$OUTPUT_ROOT"
python data_crop.py --data_name=DUT-OMRON  --data_root="$TMP_ROOT" --output_path="$OUTPUT_ROOT"
python data_crop.py --data_name=PASCAL-S  --data_root="$TMP_ROOT" --output_path="$OUTPUT_ROOT"
python data_crop.py --data_name=HKU-IS  --data_root="$TMP_ROOT" --output_path="$OUTPUT_ROOT"
python data_crop.py --data_name=DUTS-TE  --data_root="$DATA_ROOT" --output_path="$OUTPUT_ROOT"
python data_crop.py --data_name=DUTS-TR  --data_root="$DATA_ROOT" --output_path="$OUTPUT_ROOT"

# prevent wrong path
if [ -d $TMP_ROOT/SOD ]; then
  rm -rf $TMP_ROOT
fi
python sal2edge.py --data_root="$OUTPUT_ROOT/DUTS-TR/DUTS-TR-Mask/" --output_path="$OUTPUT_ROOT/DUTS-TR/DUTS-TR-Mask/" --image_list_file="$OUTPUT_ROOT/DUTS-TR/test.lst"

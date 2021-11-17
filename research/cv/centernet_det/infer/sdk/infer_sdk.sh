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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash infer_sdk.sh IMG_PATH INFER_MODE INFER_RESULT_DIR ANN_FILE"
echo "for example of inference: bash infer_sdk.sh /path/image_path infer /path/infer_result /path/annotations_path"
echo "for example of validation: bash infer_sdk.sh /path/COCO2017/val2017 eval /path/infer_result /path/COCO2017/annotations/instances_val2017.json"
echo "=============================================================================================================="
IMG_PATH=$1
INFER_MODE=$2
INFER_RESULT_DIR=$3
ANN_FILE=$4

# install nms module from third party
if python3.7 -c "import nms" > /dev/null 2>&1
then
    echo "NMS module already exits, no need reinstall."
else
    cd external || exit
    make
    python3.7 setup.py install
    cd - || exit
fi

python3.7 main.py  \
   --img_path=$IMG_PATH \
   --infer_mode=$INFER_MODE \
   --infer_result_dir=$INFER_RESULT_DIR \
   --ann_file=$ANN_FILE > infer_sdk.log 2>&1 &

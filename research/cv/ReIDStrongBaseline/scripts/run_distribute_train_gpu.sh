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

if [ $# != 5 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [DATA_DIR] [OUTPUT_PATH] [PRETRAINED_RESNET50]"
echo "for example: bash scripts/run_distribute_train_gpu.sh ./configs/market1501_config.yml 8 /path/to/dataset/ /path/to/output/ /path/to/pretrained_resnet50.pth"
echo "=============================================================================================================="
exit 1;
fi


get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
config_path=$(get_real_path "$1")

export RANK_SIZE=$2
echo $RANK_SIZE
DATA_DIR=$(get_real_path "$3")
OUTPUT_PATH=$(get_real_path "$4")
PRETRAINED_RESNET50=$(get_real_path "$5")

mpirun -n $RANK_SIZE --output-filename log_output  --allow-run-as-root --merge-stderr-to-stdout \
  python train.py  \
    --config_path="$config_path" \
    --device_target="GPU" \
    --data_dir="$DATA_DIR" \
    --ckpt_path="$OUTPUT_PATH" \
    --train_log_path="$OUTPUT_PATH" \
    --pre_trained_backbone="$PRETRAINED_RESNET50" \
    --lr_init=0.00140 \
    --lr_cri=1.0 \
    --ids_per_batch=8 \
    --is_distributed=1 > output.train.log 2>&1 &

#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

if [  $# -lt 2 ]; then
    echo "Usage: bash run_eval.sh [COCO_PATH] [CHECKPOINT_PATH]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

coco_path=$(get_real_path $1)
checkpoint=$(get_real_path $2)

BASEPATH=$(cd "`dirname $0`" || exit; pwd)

python ${BASEPATH}/../main.py --eval --output_dir=logs/Focus_DETR/R50-MS5-eval \
    --config_file=config/Focus_DETR/Focus_DETR_5scale.py --coco_path=$coco_path  \
    --resume=$checkpoint \
    --options dn_scalar=100 embed_init_tgt=TRUE \
    dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
    dn_box_noise_scale=1.0

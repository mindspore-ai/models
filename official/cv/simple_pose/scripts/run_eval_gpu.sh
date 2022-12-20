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

if [ $# -lt 3 ]
then
    echo "Usage:
          bash scripts/run_eval_gpu.sh [EVAL_MODEL_FILE] [COCO_BBOX_FILE] [DEVICE_ID]
          e.g. bash scripts/eval_gpu.sh ./ckpt/gpu_distributed/simplepose-185_1170.ckpt ./experiments/COCO_val2017_detections_AP_H_56_person.json 0
          "
exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)

cd $BASEPATH/../

if [ ! -d ./logs ];
then
    mkdir ./logs
fi

config_path="./default_config.yaml"
echo "config path is : ${config_path}"
ckpt_path="$1"
echo "evaluating model : ${ckpt_path}"
log_path="./logs/eval_gpu.log"
echo "log_path is : ${log_path}"

echo "start gpu evaluating, using device $3"
export CUDA_VISIBLE_DEVICES="$3"
nohup python eval.py --device_target GPU --eval_model_file $1 --coco_bbox_file $2 > ${log_path} 2>&1 &

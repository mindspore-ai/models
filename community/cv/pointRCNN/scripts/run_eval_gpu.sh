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

cd "dirname $0"
cd ..
export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH
if [ $# -eq 0 ]
then
    echo "No arguments supplied"
    echo "usage: ./script/eval.sh <model_path>"
    exit
fi

python eval.py --cfg_file config/default.yaml --ckpt $1 --batch_size 1 --eval_mode rcnn > eval_gpu.log 2>&1 &

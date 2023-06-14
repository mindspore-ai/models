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

echo "Please run the script as: "
echo "bash run_eval.sh DATASET_PATH"
echo "for example: bash run_eval.sh ~/DepthNet_dataset ~/Model/Ckpt/FinalCoarseNet.ckpt ~/Model/Ckpt/FinalFineNet.ckpt"
echo "After running the script, the network runs in the background, The log will be generated in eval.log"

export DATASET_PATH=$1
export COARSENET_MODEL_PATH=$2
export FINENET_MODEL_PATH=$3

cd ..
python eval.py --test_data $DATASET_PATH --coarse_ckpt_model $COARSENET_MODEL_PATH --fine_ckpt_model $FINENET_MODEL_PATH > eval.log 2>&1 &

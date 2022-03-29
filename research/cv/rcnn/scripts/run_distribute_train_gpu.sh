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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh "
echo "For example: bash run_distribute_train_gpu.sh "
echo "=============================================================================================================="

export DEVICE_NUM=8
export RANK_SIZE=8

echo "start training ... "
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -n 8 --output-filename log_output_finetune --merge-stderr-to-stdout \
python train.py  --step 0 >log_train_finetune 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -n 8 --output-filename log_output_svm --merge-stderr-to-stdout \
python train.py  --step 1 >log_train_svm 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -n 8 --output-filename log_output_regression --merge-stderr-to-stdout \
python train.py  --step 2 >log_train_regression 2>&1 &
cd ..

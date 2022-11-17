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
echo "Before running the script, you should modify 4 params of src/config.py file"
echo "including val_dataset_folder, val_gt_dir, onnx_model and device."
echo "And then, you can run the script as: bash run_onnx_eval.sh 0"
echo "=============================================================================================================="

export CUDA_VISIBLE_DEVICES="$1"
python eval_onnx.py > eval.log 2>&1 &

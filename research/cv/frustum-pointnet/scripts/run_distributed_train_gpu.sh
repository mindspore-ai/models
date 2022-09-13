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
# ==========================================================================


if [ $# != 1 ]
then
    echo "Usage: bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM]"
    echo "============================================================"
    echo "[DEVICE_NUM]: The number of cuda devices to use."
    echo "more configs can be found in `python train.py --help`"
    echo "you can edit the train parameters in this file by yourself"
    echo "============================================================"
exit 1
fi

export DEVICE_NUM=$1
export RANK_SIZE=$1
echo "start training, log will output to train_dis.log and the model file will save to log/Fpointnet*.ckpt by default"
mpirun -n $1 --allow-run-as-root --output-filename train_dis_log --merge-stderr-to-stdout python train_net.py > dis_log.txt 2>&1 &
echo 'running'

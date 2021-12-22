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
"""Functions of cells"""
import mindspore.nn as nn
from mindspore import dtype as mstype

from src.args import args


def do_keep_fp32(network, cell_types):
    """Cast cell to fp32 if cell in cell_types"""
    for name, cell in network.cells_and_names():
        if isinstance(cell, cell_types) or "features.0" in name:
            cell.to_float(mstype.float32)


def cast_amp(net):
    """cast network amp_level"""
    assert args.amp_level in ("O0", "O1", "O2", "O3")
    if args.amp_level == "O2":
        cell_types = (nn.LayerNorm, nn.Softmax, nn.BatchNorm2d, nn.BatchNorm1d, nn.GELU, nn.Sigmoid, nn.Dense,
                      nn.SyncBatchNorm)
        print(f"=> using amp_level {args.amp_level}\n"
              f"=> change {args.arch}to fp16")
        net.to_float(mstype.float16)
        do_keep_fp32(net, cell_types)
        print(f"cast {cell_types} to fp32 back")
    elif args.amp_level == "O3":
        print(f"=> using amp_level {args.amp_level}\n"
              f"=> change {args.arch} to fp16")
        net.to_float(mstype.float16)
    else:
        print(f"=> using amp_level {args.amp_level}")
        args.loss_scale = 1.
        args.is_dynamic_loss_scale = 0
        print(f"=> When amp_level is O0, using fixed loss_scale with {args.loss_scale}")

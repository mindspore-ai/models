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
"""export"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.net import SBNetWork
from src.model_utils.config import config

def run_export():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=5)

    net = SBNetWork(in_chanel=[19, 64, 128],
                    out_chanle=config.conv_channels,
                    dense_size=config.dense_sizes,
                    osize=1, lmbda=0.001,
                    isize=21, keep_prob=1.0)

    assert config.ckpt_file is not None, "config.ckpt_file is None."
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, param_dict)

    coor = Tensor(np.ones([1, 19, 21, 21, 21]), ms.float32)
    affine = Tensor(np.ones([1, 1]), ms.float32)
    inputs = [coor, affine]
    export(net, *inputs, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    run_export()

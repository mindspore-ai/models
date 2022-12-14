# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore import export
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from src.config import config as cfg
from src import Stnet_Res_model


def export_func():
    """main func"""
    # load config
    target = cfg.target
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.target)
    if target == "Ascend":
        context.set_context(device_id=cfg.device_id)
    # define net
    net = Stnet_Res_model.stnet50(input_channels=3, num_classes=cfg.class_num, T=cfg.T, N=cfg.N)
    # load pretrain model
    param_dict = load_checkpoint(cfg.ckpt_file)
    load_param_into_net(net, param_dict)
    input_data = Tensor(np.zeros([1, cfg.T, cfg.N*3, cfg.target_size, cfg.target_size]),
                        mindspore.float32)
    export(net, input_data, file_name=cfg.file_name, file_format=cfg.file_format)


if __name__ == '__main__':
    export_func()

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
"""export to mindir"""
import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from src.model_reposity import Resnet18_8s

from model_utils.config import config as cfg

context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
if cfg.device_target == "Ascend":
    context.set_context(device_id=cfg.rank)


if __name__ == "__main__":
    net = Resnet18_8s(ver_dim=cfg.vote_num * 2)
    param_dict = ms.load_checkpoint(cfg.ckpt_file)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)
    input_data = Tensor(np.zeros([1, 3, cfg.img_height, cfg.img_width]), ms.float32)
    ms.export(net, input_data, file_name=cfg.file_name, file_format=cfg.file_format)

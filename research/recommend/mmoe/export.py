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
"""export ckpt to model"""
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.mmoe import MMoE

import numpy as np
import mindspore as ms
from mindspore import context, load_checkpoint, load_param_into_net, Tensor, export

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)


def modelarts_process():
    pass


@moxing_wrapper(pre_process=modelarts_process)
def export_mmoe():
    """export MMoE"""
    net = MMoE(num_features=config.num_features, num_experts=config.num_experts, units=config.units)
    param_dict = load_checkpoint(config.ckpt_file_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([1, 499]), ms.float16)
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    export_mmoe()

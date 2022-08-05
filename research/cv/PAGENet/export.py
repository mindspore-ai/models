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

"""
##############export checkpoint file into air, mindir models#################
python export.py
"""
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.pagenet import MindsporeModel
from src.model_utils.config import config


def run_export():
    """
    run export operation
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    net = MindsporeModel(config)

    if not os.path.exists(config.ckpt_file):
        raise ValueError("config.ckpt_file is None.")
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.ones([config.batch_size, 3, 224, 224]), ms.float32)
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)


if __name__ == "__main__":
    run_export()

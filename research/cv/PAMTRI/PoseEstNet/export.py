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
"""export checkpoint file into air, mindir models"""

import argparse
import numpy as np
import mindspore
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.model import get_pose_net
from src.config import cfg, update_config

parser = argparse.ArgumentParser(description='Train PoseEstNet network')

parser.add_argument('--ckpt_path', type=str, default='')
parser.add_argument('--cfg', type=str, default='config')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend")

args = parser.parse_args()

if __name__ == '__main__':
    update_config(cfg, args)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, \
        save_graphs=False, device_id=args.device_id)

    network = get_pose_net(cfg)
    assert args.ckpt_path is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)

    input_arr = Tensor(np.zeros((32, 3, 256, 256)), mindspore.float32)

    mindspore.export(network, input_arr, file_name="PoseEstNet_export", file_format=args.file_format)

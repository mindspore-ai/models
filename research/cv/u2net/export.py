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
"""export U-2-Net model"""

import argparse

import numpy as np
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export
from src.blocks import U2NET

parser = argparse.ArgumentParser(description='checkpoint export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="u2net",
                    help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend", help="device target")
args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

if __name__ == '__main__':
    net = U2NET()
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)
    input_data = Tensor(np.zeros([1, 3, 320, 320], np.float32))
    export(net, input_data, file_name=args.file_name, file_format=args.file_format)

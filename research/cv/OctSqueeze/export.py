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

import argparse

import numpy as np
import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export

import src.network as network


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_file", "-c", type=str, default="/home/OctSqueeze/checkpoint/octsqueeze.ckpt", help="path of checkpoint"
)
parser.add_argument("--batch_size", type=int, default=98304, help="batch size of data")
parser.add_argument("--file_name", "-n", type=str, default="octsqueeze", help="name of model")
parser.add_argument("--file_format", "-f", type=str, default="MINDIR", help="format of model")
parser.add_argument(
    "--device_target",
    type=str,
    default="Ascend",
    choices=["Ascend", "GPU", "CPU"],
    help="device where the code will be implemented",
)
parser.add_argument("--device_id", type=int, default=0, help="which device where the code will be implemented")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)


def export_octsqueeze():
    """export_octsqueeze"""
    net = network.OctSqueezeNet()
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([args.batch_size, 24]), ms.float32)
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)


if __name__ == "__main__":
    # Export checkpoint in mindir format
    export_octsqueeze()

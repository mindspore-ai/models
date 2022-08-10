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
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse
import numpy as np

from mindspore import dtype as mstype
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src import SDNE, SDNEWithLossCell, SDNELoss1
from src import cfg

parser = argparse.ArgumentParser(description='Export Script')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="sdne", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
parser.add_argument('--dataset', type=str, default='WIKI',
                    choices=['WIKI', 'GRQC'])
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    config = cfg[args.dataset]

    net = SDNEWithLossCell(SDNE(config['node_size'], hidden_size=config['hidden_size']), SDNELoss1())

    assert args.ckpt_file is not None, "args.ckpt_file is None."
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)
    sdne = net.get_backbone()

    input_arr = Tensor(np.ones([args.batch_size, config['node_size']]), mstype.float32)
    export(sdne, input_arr, file_name=args.file_name, file_format=args.file_format)

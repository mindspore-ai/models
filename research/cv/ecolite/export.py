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

"""export checkpoint file into air, onnx, mindir models"""

import ast
import argparse
import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.econet import ECONet

parser = argparse.ArgumentParser(description='ECOLite export')
parser.add_argument('--dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'something', 'jhmdb'])
parser.add_argument('--modality', default="RGB", type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('--arch', type=str, default="ECO")
parser.add_argument('--num_segments', type=int, default=4)
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--consensus_type', type=str, default='identity',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--dropout', '--do', default=0.7, type=float,
                    metavar='DO', help='dropout ratio (default: 0.7)')
parser.add_argument('--no_partialbn', '--npb', default=True)
parser.add_argument("--device_id", type=int, default=0, help="device id")
parser.add_argument("--checkpoint_path", default="/path/to/ms_model_kinetics_checkpoint1020.ckpt", type=str,
                    required=True, help="checkpoint file path.")
parser.add_argument("--file_name", type=str, default="ecolite", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend"], default="Ascend", help="device target")
args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)
if args.dataset == 'ucf101':
    num_class = 101
    rgb_read_format = "{:06d}.jpg"
elif args.dataset == 'hmdb51':
    num_class = 51
    rgb_read_format = "{:05d}.jpg"
elif args.dataset == 'kinetics':
    num_class = 400
    rgb_read_format = "{:04d}.jpg"
elif args.dataset == 'something':
    num_class = 174
    rgb_read_format = "{:05d}.jpg"

if __name__ == '__main__':
    param_dict = load_checkpoint(args.checkpoint_path)
    net = ECONet(num_class, args.num_segments, args.modality,
                 base_model=args.arch,
                 consensus_type=args.consensus_type, dropout=args.dropout,
                 partial_bn=not ast.literal_eval(args.no_partialbn))
    net.set_train(False)
    load_param_into_net(net, param_dict)
    batch_size = args.batch_size
    data = np.random.uniform(0.0, 1.0, size=[batch_size, args.num_segments * 3, 224, 224]).astype(np.float32)

    export(net, Tensor(data), file_name=args.file_name, file_format=args.file_format)

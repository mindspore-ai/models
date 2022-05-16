# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""export mindir,air or onnx format"""
import argparse

import numpy as np

import mindspore
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

import src.genotypes as genotypes
from src.model import NetworkCIFAR as Network

parser = argparse.ArgumentParser(description='PDarts export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--testing_shape", type=int, default=32, help="test shape")
parser.add_argument('--num_classes', default=10,
                    type=int, help='number of classes')

parser.add_argument('--init_channels', type=int,
                    default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20,
                    help='total number of layers')
parser.add_argument('--auxiliary', action='store_true',
                    default=True, help='use auxiliary tower')
parser.add_argument('--arch', type=str, default='PDARTS',
                    help='which architecture to use')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.3, help='drop path probability')
parser.add_argument('--train_epochs', type=int, default=600,
                    help='num of training epochs')
parser.add_argument("--train_batch_size", type=int,
                    default=128, help="batch size")

parser.add_argument("--ckpt_file", type=str, required=True,
                    help="Checkpoint file path.")
parser.add_argument("--file_name", type=str,
                    default="PDarts", help="output file name.")
parser.add_argument('--file_format', type=str,
                    choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend"], default="Ascend",
                    help="device target")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    device_target=args.device_target, device_id=args.device_id)

if __name__ == "__main__":
    ts_shape = args.testing_shape
    CIFAR_CLASSES = 10

    print(genotypes.Genotype)
    if args.arch == 'PDARTS':
        genotype = genotypes.PDARTS
    print('---------Genotype---------')
    print(genotype)
    print('--------------------------')
    network = Network(args.init_channels, CIFAR_CLASSES,
                      args.layers, args.auxiliary, genotype)
    network.training = False
    network.drop_path_prob = args.drop_path_prob * 300 / args.train_epochs
    keep_prob = 1. - network.drop_path_prob
    epoch_mask = []
    for i in range(args.layers):
        layer_mask = []
        for j in range(5 * 2):
            mask = np.array([np.random.binomial(1, p=keep_prob)
                             for k in range(args.train_batch_size)])
            mask = mask[:, np.newaxis, np.newaxis, np.newaxis]
            mask = Tensor(mask, mindspore.float16)
            layer_mask.append(mask)
        epoch_mask.append(layer_mask)
    network.epoch_mask = epoch_mask

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)

    input_data = Tensor(
        np.zeros([args.batch_size, 3, ts_shape, ts_shape]), mindspore.float32)

    export(network, input_data, file_name=args.file_name,
           file_format=args.file_format)
    print('===============export finished...================')

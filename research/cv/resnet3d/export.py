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
"""
export AIR or MINDIR model.
"""
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, context, export
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.ResNet3D import generate_model

parser = argparse.ArgumentParser(description='ResNet3D_export')
parser.add_argument('--device_id', type=int, default=0, help='Device id.')
parser.add_argument('--ckpt_file', type=str, required=True,
                    help='Checkpoint file path.')
parser.add_argument('--file_name', type=str,
                    default='resnet-3d', help='Output file name.')
parser.add_argument('--file_format', type=str,
                    choices=['AIR', 'MINDIR', 'ONNX'], default='MINDIR', help='File format.')
parser.add_argument('--device_target', type=str, choices=['Ascend', 'CPU', 'GPU'], default='Ascend',
                    help='Device target')
parser.add_argument('--sample_duration', type=int, default=16)
parser.add_argument('--sample_size', type=int, default=112)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_classes', type=int, choices=[51, 101])
args = parser.parse_args()


class NetWithSoftmax(nn.Cell):
    """
    Add Softmax module to network.
    """

    def __init__(self, network):
        super(NetWithSoftmax, self).__init__()
        self.softmax = nn.Softmax()
        self.net = network

    def construct(self, x):
        out = self.net(x)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    if args.device_target == 'Ascend':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.device_target, device_id=args.device_id)
    net = generate_model(n_classes=args.n_classes, no_max_pool=False)
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    net = NetWithSoftmax(net)
    net.set_train(False)

    input_data = Tensor(np.zeros(
        [args.batch_size, 3, args.sample_duration, args.sample_size, args.sample_size]), mindspore.float32)
    print(input_data.shape)
    export(net, input_data, file_name=args.file_name,
           file_format=args.file_format)

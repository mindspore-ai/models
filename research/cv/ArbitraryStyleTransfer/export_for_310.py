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

"""file for evaling"""
import argparse
import numpy as np

import mindspore
from mindspore import Tensor, Parameter
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.serialization import export
from src.inceptionv3 import inceptionv3
from src.model import get_model

set_seed(1)
parser = argparse.ArgumentParser(description="style transfer train")
# data loader
parser.add_argument("--inception_ckpt", type=str, default='./pretrained_model/inceptionv3.ckpt')
parser.add_argument('--file_format', type=str, choices=['AIR', 'ONNX', 'MINDIR'], default='MINDIR', \
                    help='file format')
parser.add_argument("--ckpt_path", type=str, default='./ckpt/style_transfer_model_0100.ckpt')
parser.add_argument('--platform', type=str, choices=['Ascend', 'GPU'], default='Ascend', help='Ascend or GPU')
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument("--image_size", type=int, default=256, help='image size, default: image_size.')
parser.add_argument('--file_name', type=str, default='style_transfer', help='output file name prefix.')
parser.add_argument("--style_dim", type=int, default=100,
                    help="Style vector dimension. default: 100")
parser.add_argument('--init_type', type=str, default='normal', choices=("normal", "xavier"), \
                    help='network initialization, default is normal.')
parser.add_argument('--init_gain', type=float, default=0.02, \
                    help='scaling factor for normal, xavier and orthogonal, default is 0.02.')

if __name__ == '__main__':
    args = parser.parse_args()
    image_size = args.image_size
    transfer_net = get_model(args)
    params = load_checkpoint(args.ckpt_path)
    load_param_into_net(transfer_net, params)


    class stylization(nn.Cell):
        def __init__(self):
            super(stylization, self).__init__()
            self.inception = inceptionv3(args.inception_ckpt)
            self.transfer_net = transfer_net
            self.scale = Parameter(Tensor(np.array([2, 2, 2]).reshape([1, 3, 1, 1]), mindspore.float32))
            self.offset = Parameter(Tensor(np.array([1, 1, 1]).reshape([1, 3, 1, 1]), mindspore.float32))
            self.op_concat = ops.Concat(axis=0)

        def construct(self, content, style_shifted):
            s_in_feat = self.inception(style_shifted)
            c_in_feat = self.inception(content[1])
            interporated_stylied_img = self.transfer_net.construct_interpolation_310(content[0], s_in_feat, c_in_feat)
            return self.op_concat([style_shifted, interporated_stylied_img, content[0]])


    model = stylization()
    input1_shape = [2, 1, 3, args.image_size, args.image_size]
    input2_shape = [1, 3, args.image_size, args.image_size]
    input_array_1 = Tensor(np.random.uniform(-1.0, 1.0, size=input1_shape).astype(np.float32))
    input_array_2 = Tensor(np.random.uniform(-1.0, 1.0, size=input2_shape).astype(np.float32))
    G_file = f"{args.file_name}_model"
    export(model, input_array_1, input_array_2, file_name=G_file, file_format=args.file_format)

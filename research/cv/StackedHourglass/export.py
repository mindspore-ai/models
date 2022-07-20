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
"""export model"""
import os
import numpy as np
from src.config import parse_args
from src.models.StackedHourglassNet import StackedHourglassNet
import src.dataset.MPIIDataLoader as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, export, load_checkpoint, load_param_into_net
args = parse_args()

class MaxPool2dFilter(nn.Cell):
    """
        Maxpool 2d for filter
    """

    def __init__(self):
        super(MaxPool2dFilter, self).__init__()
        self.pool = nn.MaxPool2d(3, 1, "same")
        self.eq = ops.Equal()

    def construct(self, x):
        """
            forward
        """
        maxm = self.pool(x)
        return self.eq(maxm, x)


class Hourglass(nn.Cell):
    """
        Hourglass
    """

    def __init__(self, network):
        super(Hourglass, self).__init__(auto_prefix=False)
        self.net = network
        self.pool = nn.MaxPool2d(3, 1, "same")
        self.eq = ops.Equal()


    def construct(self, x):
        """
            forward
        """
        tmp1 = self.net(x)
        tmp2 = self.net(x[:, ::-1])
        tmp = ops.Concat(0)((tmp1, tmp2))

        det = tmp[0, -1] + tmp[1, -1, :, :, ::-1][ds.flipped_parts["mpii"]]

        det = det / 2

        det = ops.minimum(det, 1)
        det0 = det
        det = ops.expand_dims(det, 0)
        maxm = self.pool(det)
        maxm = self.eq(maxm, det)
        det1 = det * maxm
        return det0, det1


if __name__ == "__main__":
    if not os.path.exists(args.ckpt_file):
        print("ckpt file not valid")
        exit()

    # Import net
    net = StackedHourglassNet(args.nstack, args.inp_dim, args.oup_dim)
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([1, args.input_res, args.input_res, 3], np.float32))
    export(net, input_arr, file_name='Hourglass', file_format=args.file_format)

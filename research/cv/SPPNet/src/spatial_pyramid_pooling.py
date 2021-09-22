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
"""pool"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P


class SpatialPyramidPool(nn.Cell):
    """
    SpatialPyramidPool
    """
    def __init__(self, previous_conv_size=13, out_pool_size=None):
        '''
        args:
            previous_conv_size(int): input feature map size
            out_pool_size(tuple): output pooling size
            e.g: input_size: (6, 3, 2, 1) out_pool_size=(6*6+3*3+2*2+1*1) * batch_size
        '''
        super(SpatialPyramidPool, self).__init__()

        self.previous_conv_size = previous_conv_size
        self.out_pool_size = out_pool_size
        self.cat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.maxpool_1x1 = ops.ReduceMax(keep_dims=True)
        self.padding = P.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))

    def construct(self, x):
        """
        input: x (last_out_channels ,previous_conv_size,previous_conv_size)

        output: x (Vector)
        """
        B, _, _, _ = ops.Shape()(x)
        spp = None
        for pool_count in range(len(self.out_pool_size)):
            size = self.previous_conv_size / self.out_pool_size[pool_count]

            if size > size // 1:
                size = size // 1 + 1
            else:
                size = size // 1

            stride = self.previous_conv_size / self.out_pool_size[pool_count]
            stride = stride // 1

            if self.out_pool_size[pool_count] == 1:
                spp_temp = self.maxpool_1x1(x, (2, 3))
            elif self.out_pool_size[pool_count] == 6 and self.previous_conv_size == 10:
                x_pad = self.padding(x)
                spp_temp = nn.MaxPool2d(2, 2, "valid")(x_pad)
            else:
                spp_temp = nn.MaxPool2d(size, stride, "valid")(x)

            if pool_count == 0:
                spp = self.reshape(spp_temp, (B, -1))
            else:
                spp = self.cat((spp, self.reshape(spp_temp, (B, -1))))

        return spp

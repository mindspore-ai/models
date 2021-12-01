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
# ===========================================================================
"""Schedule drop path method"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer


class DropPath(nn.Cell):
    """DropPath"""
    def __init__(self, keep_prob, cell_num, total_num_cells, total_steps):
        super(DropPath, self).__init__()

        if keep_prob > 1.0 or keep_prob <= 0.0:
            raise ValueError('keep_prob must in (0, 1]')

        self.cast = ops.Cast()
        self.scast = ops.ScalarCast()

        self.keep_prob = self.scast(keep_prob, mindspore.float32)
        self.cell_num = self.scast(cell_num, mindspore.float32)
        self.total_num_cells = self.scast(total_num_cells, mindspore.float32)
        self.global_step = Parameter(initializer(0, [], mindspore.int32), requires_grad=False)
        self.total_steps = self.scast(total_steps, mindspore.float32)

        self.min = ops.Minimum()
        self.uniform = ops.UniformReal()
        self.floor = ops.Floor()
        self.assignadd = ops.AssignAdd()

    def construct(self, net):
        """construct"""
        drop_path_keep_prob = self.keep_prob
        if self.training and (drop_path_keep_prob < 1.0):
            # Scale keep prob by layer number.
            layer_ratio = (self.cell_num + 1.0) / self.total_num_cells
            drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
            # Decrease keep prob over time.
            current_step = self.cast(self.assignadd(self.global_step, 1), mindspore.float32)
            current_ratio = current_step / self.total_steps
            drop_path_keep_prob = 1.0 - current_ratio * (1.0 - drop_path_keep_prob)
            # Drop path.
            noise_shape = (net.shape[0], 1, 1, 1)
            random_tensor = drop_path_keep_prob + self.uniform(noise_shape)
            binary_tensor = self.cast(self.floor(random_tensor), mindspore.float32)
            keep_prob_inv = self.cast(1.0 / drop_path_keep_prob, mindspore.float32)
            net = net * keep_prob_inv * binary_tensor
        return net

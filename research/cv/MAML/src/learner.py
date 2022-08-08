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

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import initializer as init
from src.KaimingNormal import KaimingNormal

class reduce_mean(nn.Cell):
    def __init__(self):
        super(reduce_mean, self).__init__()
        self.mean_op = ops.ReduceMean(keep_dims=False)
    def construct(self, x):
        return self.mean_op(x, (2, 3))

class Learner(nn.Cell):
    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.ones = ops.Ones()
        self.zeros = ops.Zeros()
        self.stack = ops.Stack()
        self.config = config

        CellList = []
        for _, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                cell = nn.Conv2d(in_channels=param[0], out_channels=param[1], kernel_size=(param[2], param[3]),
                                 stride=param[4], padding=param[5], has_bias=True, pad_mode=param[6])
            elif name == 'bn':
                cell = nn.BatchNorm2d(num_features=param[0], momentum=0.99, eps=0.001).set_train(True)
            elif name == 'relu':
                cell = nn.ReLU()
            elif name == 'reduce_mean':
                cell = reduce_mean()
            elif name == 'linear':
                cell = nn.Dense(*param)
            elif name == 'convt2d':
                cell = nn.Conv2dTranspose(*param[:4])
            else:
                raise NotImplementedError
            CellList.append(cell)
        self.net = nn.SequentialCell(CellList)
        self._initialize_weights()

    def construct(self, x):

        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        return self.net(x)

    def _initialize_weights(self):
        np.random.seed(123)
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Conv2d)):
                cell.weight.set_data(init.initializer(KaimingNormal(mode='fan_out'),
                                                      cell.weight.data.shape,
                                                      cell.weight.data.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.data.shape,
                                                        cell.bias.data.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(KaimingNormal(mode='fan_out'),
                                                      cell.weight.data.shape,
                                                      cell.weight.data.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.data.shape,
                                                        cell.bias.data.dtype))
            elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
                cell.gamma.set_data(init.initializer('ones', cell.gamma.data.shape))
                cell.beta.set_data(init.initializer('zeros', cell.beta.data.shape))
            elif isinstance(cell, nn.Conv2dTranspose):
                cell.weight.set_data(init.initializer(KaimingNormal(mode='fan_out'),
                                                      cell.weight.data.shape,
                                                      cell.weight.data.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.data.shape,
                                                        cell.bias.data.dtype))

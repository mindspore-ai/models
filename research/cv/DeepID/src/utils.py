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
"""Dataset distributed sampler."""
from __future__ import division
import math
import numpy as np

import mindspore.nn as nn
from mindspore.common import initializer as init, set_seed

from src.model import DeepID
from src.loss import cal_acc

set_seed(1)
np.random.seed(1)


class DistributedSampler:
    """Distributed sampler."""
    def __init__(self, dataset_size, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            print("***********Setting world_size to 1 since it is not passed in ******************")
            num_replicas = 1
        if rank is None:
            print("***********Setting rank to 0 since it is not passed in ******************")
            rank = 0

        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.epoch = 0
        self.rank = rank
        self.num_samples = int(math.ceil(dataset_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # shuffle based on epoch
        if self.shuffle:
            indices = np.random.RandomState(seed=self.epoch).permutation(self.dataset_size)
            indices = indices.tolist()
            self.epoch += 1
        else:
            indices = list(range(self.dataset_size))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.

    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'KaimingUniform':
                cell.weight.set_data(init.initializer(init.HeUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            elif init_type == 'truncatedNormal':
                cell.weight.set_data(init.initializer(init.TruncatedNormal(), cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, nn.GroupNorm):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))



def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.trainable_params():
        num_params += np.prod(p.shape)
    print(model)
    print(name)
    print('The number of parameters: {}'.format(num_params))


def get_network(args, num_class, feature=False):
    """Create and initial a generator and a discriminator."""

    deepID = DeepID(args.input_dim, num_class, feature)

    init_weights(deepID, 'truncatedNormal', math.sqrt(5))

    print_network(deepID, 'DeepID')

    return deepID

def eval_func(net, img, label):
    predict = net(img)
    acc = cal_acc(predict, label)
    return acc

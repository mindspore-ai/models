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

"""tools"""
import os
import numpy as np
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore.common import initializer as ini
from model_utils.config import config


class Identity(nn.Cell):
    def construct(self, inputvar):
        """forward"""
        return inputvar


class SegmentConsensus(nn.Cell):
    """SegmentConsensus"""

    def __init__(self, consensus_type, dims=1):
        super().__init__()
        self.consensus_type = consensus_type
        self.dims = dims
        self.shape = None

    def construct(self, input_tensor):
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dims, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(nn.Cell):
    """ConsensusModule"""

    def __init__(self, consensus_type, dims=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dims = dims
        self.segment_consensus = SegmentConsensus(self.consensus_type, self.dims)

    def construct(self, inputvar):
        """construct"""
        return self.segment_consensus(inputvar)


def load_pretrain_checkpint(model_dict, net):
    """load_pretrain_checkpint"""
    if config.resume:
        pretrained_dict = load_checkpoint(config.resume)
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint(fintune) '{}'".format(config.resume)))
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            if (k in model_dict) and (v.shape == model_dict[k].shape):
                new_state_dict[k] = v
        un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
        print("un_init_dict_keys:", len(un_init_dict_keys))
        for k in un_init_dict_keys:
            shape = model_dict[k].shape
            new_state_dict[k] = Parameter(Tensor(np.zeros(shape), dtype=mstype.float32))
            if 'weight' in k:
                shape = new_state_dict[k].shape
                new_state_dict[k].set_data(ini.initializer('xavier_uniform', shape))
            elif 'bias' in k:
                shape = new_state_dict[k].shape
                new_state_dict[k].set_data(ini.initializer('zeros', shape))
            load_param_into_net(net, new_state_dict)
    else:
        print("=> Train from scratch")

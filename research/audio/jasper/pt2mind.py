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


import argparse
import re

import numpy as np
from sympy import arg
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

parser = argparse.ArgumentParser(description='pth translate to ckpt')
parser.add_argument('--pth', type='str',
                    default='/data/Jasper_epoch10_checkpoint.pt', help='path of pth')

args = parser.parse_args()


def convert_v1_state_dict(state_dict):
    rules = [
        ('^jasper_encoder.encoder.', 'encoder.layers.'),
        ('^jasper_decoder.decoder_layers.', 'decoder.layers.'),
    ]
    ret = {}
    for k, v in state_dict.items():
        if k.startswith('acoustic_model.'):
            continue
        if k.startswith('audio_preprocessor.'):
            continue
        for pattern, to in rules:
            k = re.sub(pattern, to, k)
        ret[k] = v

    return ret


checkpoint = torch.load(arg.pth, map_location="cpu")

state_dic = convert_v1_state_dict(checkpoint['state_dict'])


mydict = state_dic
newparams_list = []
names = [item for item in mydict if 'num_batches_tracked' not in item]
i = 0
for name in names:
    parameter = mydict[name].numpy()
    param_dict = {}

    if i % 5 == 0:
        name = name.replace('weight', 'conv1.weight')
        parameter = np.expand_dims(parameter, axis=2)
    elif i % 5 == 1:
        name = name.replace('weight', 'batchnorm.gamma')
    elif i % 5 == 2:
        name = name.replace('bias', 'batchnorm.beta')
    elif i % 5 == 3:
        name = name.replace('running_mean', 'batchnorm.moving_mean')
    else:
        name = name.replace('running_var', 'batchnorm.moving_variance')

    if i == 540:
        name = name.replace('0.conv1.weight', 'weight')
    if i == 541:
        name = name.replace('0.bias', 'bias')

    param_dict['name'] = name
    param_dict['data'] = Tensor(parameter)
    newparams_list.append(param_dict)
    if i % 5 == 4:
        newparams_list[i-3], newparams_list[i-2], newparams_list[i-1], newparams_list[i] = \
            newparams_list[i - 1], newparams_list[i], newparams_list[i -
                                                                     3], newparams_list[i-2]

    i += 1

save_checkpoint(newparams_list, './jasper_mindspore_10.ckpt')
print("end")

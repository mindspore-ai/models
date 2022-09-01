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
"""
The utils of SRFlow
"""

from collections import OrderedDict
import yaml
from yaml import Loader, Dumper
import torch

from mindspore import Tensor
from mindspore import save_checkpoint, load_checkpoint


def OrderedYaml():
    """

    Returns: yaml

    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)

    return Loader, Dumper


def train2test(path, testpath):
    """

    Args:
        path:  the file of train ckpt
        testpath: the file of test ckpt

    Returns: save test ckpt

    """
    param_dict_1 = load_checkpoint(path)

    new_params_list = []
    for key, value in param_dict_1.items():
        param_dict = {}
        if 'invconv' in key:
            value = value.asnumpy()
            value = torch.Tensor(value)
            value = torch.inverse(value.double()).float().view((value.shape[0], value.shape[1], 1, 1))
            value = value.reshape((value.shape[0], value.shape[1]))
            value = value.numpy()
        if "UpsamplerNet.layers" in key:
            if key[31] == '1':
                if key[34] == '.':
                    key = key[:33] + str(19 - int(key[33])) + key[34:]
                else:
                    key = key[:33] + str(19 - int(key[33:35])) + key[35:]
            elif key[31] == '2':
                if key[34] == '.':
                    key = key[:33] + str(18 - int(key[33])) + key[34:]
                else:
                    key = key[:33] + str(18 - int(key[33:35])) + key[35:]
            elif key[31] == '3':
                if key[34] == '.':
                    key = key[:33] + str(18 - int(key[33])) + key[34:]
                else:
                    key = key[:33] + str(18 - int(key[33:35])) + key[35:]

        value = Tensor(value)
        param_dict['name'] = key
        param_dict['data'] = value
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, testpath)


def pth2ckpt(pth_path, train_ckpt_path, test_ckpt_path, mode):
    '''
    pth -> ckpt
    mode:"train" or "test"
    '''
    param_dict = torch.load(pth_path)
    new_params_list = []
    for key, value in param_dict.items():
        param_dict = {}
        key = 'SRFlow.' + key
        value = value.numpy()

        if "UpsamplerNet.layers." in key:
            i = key[30:]
            i = i[:4]
            for j in range(0, 20):
                if '.{}.'.format(str(j)) in i:
                    key = key[:30] + '_1' + key[30:]
                    if key[34] == '.':
                        key = key[:33] + str(int(key[33])) + key[34:]
                    else:
                        key = key[:33] + str(int(key[33:35])) + key[35:]
            for j in range(20, 39):
                if '.{}.'.format(str(j)) in i:
                    key = key[:30] + '_2' + key[30:]
                    if key[34] == '.':
                        key = key[:33] + str(int(key[33]) - 20) + key[34:]
                    else:
                        key = key[:33] + str(int(key[33:35]) - 20) + key[35:]
            for j in range(39, 58):
                if '.{}.'.format(str(j)) in i:
                    key = key[:30] + '_3' + key[30:]
                    if key[34] == '.':
                        key = key[:33] + str(int(key[33]) - 39) + key[34:]
                    else:
                        key = key[:33] + str(int(key[33:35]) - 39) + key[35:]

        value = Tensor(value)
        param_dict['name'] = key
        param_dict['data'] = value
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list,  train_ckpt_path)
    if mode == "test":
        train2test(train_ckpt_path, test_ckpt_path)


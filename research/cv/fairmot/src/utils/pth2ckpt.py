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
pth to ckpt
"""
import torch
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint


def pth2ckpt(path='fairmot_dla34.pth'):
    """pth to ckpt """
    par_dict = torch.load(path, map_location='cpu')['state_dict']
    new_params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        new_name = name
        new_name = new_name.replace('level0', 'level0.0', 1)
        new_name = new_name.replace('level1', 'level1.0', 1)
        new_name = new_name.replace('hm', 'hm_fc', 1)
        new_name = new_name.replace('ida_0', 'ida_nfs.0', 1)
        new_name = new_name.replace('ida_1', 'ida_nfs.1', 1)
        new_name = new_name.replace('ida_2', 'ida_nfs.2', 1)
        new_name = new_name.replace('proj_1', 'proj.0', 1)
        new_name = new_name.replace('proj_2', 'proj.1', 1)
        new_name = new_name.replace('proj_3', 'proj.2', 1)
        new_name = new_name.replace('up_1', 'up.0', 1)
        new_name = new_name.replace('up_2', 'up.1', 1)
        new_name = new_name.replace('up_3', 'up.2', 1)
        new_name = new_name.replace('node_1', 'node.0', 1)
        new_name = new_name.replace('node_2', 'node.1', 1)
        new_name = new_name.replace('node_3', 'node.2', 1)
        new_name = new_name.replace('id.0', 'id_fc.0', 1)
        new_name = new_name.replace('id.2', 'id_fc.2', 1)
        new_name = new_name.replace('reg', 'reg_fc', 1)
        new_name = new_name.replace('wh', 'wh_fc', 1)
        new_name = new_name.replace('bn.running_mean', 'bn.moving_mean', 1)
        new_name = new_name.replace('bn.running_var', 'bn.moving_variance', 1)
        if new_name.startswith('base'):
            if new_name.endswith('.0.weight'):
                new_name = new_name[:new_name.rfind('0.weight')]
                new_name = new_name + 'conv.weight'
            if new_name.endswith('.1.weight'):
                new_name = new_name[:new_name.rfind('1.weight')]
                new_name = new_name + 'batchnorm.gamma'
            if new_name.endswith('.1.bias'):
                new_name = new_name[:new_name.rfind('1.bias')]
                new_name = new_name + 'batchnorm.beta'
            if new_name.endswith('.1.running_mean'):
                new_name = new_name[:new_name.rfind('1.running_mean')]
                new_name = new_name + 'batchnorm.moving_mean'
            if new_name.endswith('.1.running_var'):
                new_name = new_name[:new_name.rfind('1.running_var')]
                new_name = new_name + 'batchnorm.moving_variance'
            if new_name.endswith('conv1.weight'):
                new_name = new_name[:new_name.rfind('conv1.weight')]
                new_name = new_name + 'conv_bn_act.conv.weight'
            if new_name.endswith('bn1.weight'):
                new_name = new_name[:new_name.rfind('bn1.weight')]
                new_name = new_name + 'conv_bn_act.batchnorm.gamma'
            if new_name.endswith('bn1.bias'):
                new_name = new_name[:new_name.rfind('bn1.bias')]
                new_name = new_name + 'conv_bn_act.batchnorm.beta'
            if new_name.endswith('bn1.running_mean'):
                new_name = new_name[:new_name.rfind('bn1.running_mean')]
                new_name = new_name + 'conv_bn_act.batchnorm.moving_mean'
            if new_name.endswith('bn1.running_var'):
                new_name = new_name[:new_name.rfind('bn1.running_var')]
                new_name = new_name + 'conv_bn_act.batchnorm.moving_variance'
            if new_name.endswith('conv2.weight'):
                new_name = new_name[:new_name.rfind('conv2.weight')]
                new_name = new_name + 'conv_bn.conv.weight'
            if new_name.endswith('bn2.weight'):
                new_name = new_name[:new_name.rfind('bn2.weight')]
                new_name = new_name + 'conv_bn.batchnorm.gamma'
            if new_name.endswith('bn2.bias'):
                new_name = new_name[:new_name.rfind('bn2.bias')]
                new_name = new_name + 'conv_bn.batchnorm.beta'
            if new_name.endswith('bn2.running_mean'):
                new_name = new_name[:new_name.rfind('bn2.running_mean')]
                new_name = new_name + 'conv_bn.batchnorm.moving_mean'
            if new_name.endswith('bn2.running_var'):
                new_name = new_name[:new_name.rfind('bn2.running_var')]
                new_name = new_name + 'conv_bn.batchnorm.moving_variance'
            new_name = new_name.replace('bn.weight', 'bn.gamma', 1)
            new_name = new_name.replace('bn.bias', 'bn.beta', 1)
        new_name = new_name.replace('actf.0.weight', 'actf.0.gamma', 1)
        new_name = new_name.replace('actf.0.bias', 'actf.0.beta', 1)
        new_name = new_name.replace('actf.0.running_mean', 'actf.0.moving_mean', 1)
        new_name = new_name.replace('actf.0.running_var', 'actf.0.moving_variance', 1)
        param_dict['name'] = new_name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)
        print(f"{name}\t -> \t{new_name}")
    save_checkpoint(new_params_list, '{}_ms.ckpt'.format(path[:path.rfind('.pth')]))


pth2ckpt('crowdhuman_dla34.pth')

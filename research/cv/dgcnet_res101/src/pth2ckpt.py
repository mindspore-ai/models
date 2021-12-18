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
python pth2ckpt.py
"""
import argparse
import os
import os.path as osp
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor


def get_arguments():
    """
    Parse all the arguments
    Returns: args
    A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="pth2ckpt")
    parser.add_argument("--load_path", type=str, default="./pretrain/resnet101-deep.pth",
                        help="Path to the pytorch model.")
    parser.add_argument("--save_path", type=str, default="./pretrain",
                        help="Path to the directory containing the converted model.")
    args = parser.parse_args()
    return args

def pytorch2mindspore():
    """

    Returns:
        object:
    """
    check_list = ['net.conv1.1.weight', 'net.conv1.1.bias', 'net.conv1.1.running_mean', 'net.conv1.1.running_var',
                  'net.conv1.4.weight', 'net.conv1.4.bias', 'net.conv1.4.running_mean', 'net.conv1.4.running_var']
    args = get_arguments()
    # make save dir
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # load pretrain params
    print("Load model from pth path ... ")
    par_dict = torch.load(args.load_path, map_location='cpu')

    new_params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]

        if name not in ['fc.weight', 'fc.bias']:
            name = 'net.' + name
            if name.find('.bn') != -1 or name.find('downsample.1.') != -1 or name in check_list:
                name = name.replace('weight', 'gamma')
                name = name.replace('bias', 'beta')
                name = name.replace('running_mean', 'moving_mean')
                name = name.replace('running_var', 'moving_variance')

        print('========================converted_name', name)

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())

        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, osp.join(args.save_path, 'dgcnet_deep.ckpt'))


pytorch2mindspore()

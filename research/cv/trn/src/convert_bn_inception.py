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
"""Convert BNInception weights from the torch model to MindSpore format."""
import argparse

import torch
from mindspore import Tensor
from mindspore import save_checkpoint, context

_BN_PARAMS_RENAME_MAP = {
    "bias": "beta",
    "weight": "gamma",
    "running_mean": "moving_mean",
    "running_var": "moving_variance"
}

_INCEPTION_BLOCK_PREFIX = 'inception_'
_INCEPTION_PREFIX_POS = len(_INCEPTION_BLOCK_PREFIX) + 2
_BN_LABEL = '_bn.'


def torch_to_ms_tensor(torch_parameter: torch.nn.Parameter, parameter_name: str) -> Tensor:
    """Convert tensor data to MindSpore format"""
    numpy_data = torch_parameter.cpu().float().numpy()
    if '_bn' in parameter_name:
        numpy_data = numpy_data.reshape(-1)
    return Tensor(numpy_data)


def rename_parameter(parameter_name: str) -> str:
    """Rename a single parameter"""
    if parameter_name.startswith('fc.'):
        return parameter_name

    if parameter_name.startswith(_INCEPTION_BLOCK_PREFIX):
        parameter_name = f'{parameter_name[:_INCEPTION_PREFIX_POS]}.branch{parameter_name[_INCEPTION_PREFIX_POS:]}'

    # Handle batch normalization layers
    bn_pos = parameter_name.find(_BN_LABEL)
    if bn_pos != -1:
        param_name_suffix = parameter_name[bn_pos + len(_BN_LABEL):]
        new_param_suffix = _BN_PARAMS_RENAME_MAP[param_name_suffix]
        return f'{parameter_name[:bn_pos]}.bn.{new_param_suffix}'

    # Handle convolution layers
    param_name_parts = parameter_name.rsplit('.', 1)
    return f'{param_name_parts[0]}.conv.{param_name_parts[1]}'


def convert_torch_checkpoint_to_mindspore(torch_state_dict: dict) -> dict:
    """Perform conversion"""
    ms_model_parameters = {
        rename_parameter(name): torch_to_ms_tensor(param, name)
        for name, param in torch_state_dict.items()
    }

    return ms_model_parameters


def save_ms_model_parameters(model_parameters: dict, save_path: str):
    """Save MindSpore checkpoint"""
    data_to_save = [
        {'name': name, 'data': param}
        for name, param in model_parameters.items()
    ]
    save_checkpoint(data_to_save, save_path)


def convert():
    """Run conversion"""
    parser = argparse.ArgumentParser(description="BN Inception's weights converter from torch to mindspore")

    parser.add_argument('--torch_ckpt_path', type=str, required=True)
    parser.add_argument('--ms_ckpt_out_path', type=str, required=True)
    args = parser.parse_args()

    context.set_context(mode=context.PYNATIVE_MODE)

    ms_ckpt_data = convert_torch_checkpoint_to_mindspore(torch.load(args.torch_ckpt_path))
    save_ms_model_parameters(ms_ckpt_data, args.ms_ckpt_out_path)


if __name__ == '__main__':
    convert()

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
"""Convert the VGG backbone from the PyTorch format to the MindSpore format"""
import argparse
from pathlib import Path

import torch
from mindspore import Tensor
from mindspore import context
from mindspore import save_checkpoint

from src.model import VGG as ms_vgg

_VGG_PARAM = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
_IN_CHANNELS = 3


def pytorch_vgg(cfg, i, batch_norm=False):
    """Build VGG model"""
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = torch.nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = torch.nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               torch.nn.ReLU(inplace=True), conv7, torch.nn.ReLU(inplace=True)]
    return layers


def convert():
    """Convert the checkpoint from PyTorch to the MindSpore format"""
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_path", type=str, required=True, help="Path to pre-trained VGG backbone")
    args = parser.parse_args()

    backbone_path = Path(args.backbone_path).resolve()

    if not backbone_path.exists():
        raise ValueError('Pre-trained VGG backbone does not exist')

    ms_model = ms_vgg(_VGG_PARAM, _IN_CHANNELS, batch_norm=False, pretrained=None)
    pt_model = torch.nn.ModuleList(pytorch_vgg(_VGG_PARAM, _IN_CHANNELS, batch_norm=False))
    weights = torch.load(args.backbone_path)
    pt_model.load_state_dict(weights)

    pt_names = {}
    for name, param in pt_model.named_parameters():
        pt_names['layers.' + name] = param.cpu().detach().numpy()

    for param in ms_model.trainable_params():
        param.set_data(Tensor(pt_names[param.name]))

    converted_ckpt_path = backbone_path.with_suffix('.ckpt')

    print(args.backbone_path)
    print(converted_ckpt_path)
    save_checkpoint(ms_model, str(converted_ckpt_path))
    print(f'Succesfully converted VGG backbone. Path to checkpoint: {converted_ckpt_path}')


if __name__ == '__main__':
    convert()

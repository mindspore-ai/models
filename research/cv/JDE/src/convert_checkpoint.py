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
"""Checkpoint import."""
from pathlib import Path

import torch
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import save_checkpoint

from cfg.config import config
from src.darknet import DarkNet
from src.darknet import ResidualBlock


def convert(cfg):
    """
    Init the DarkNet53 model, load PyTorch checkpoint,
    change the keys order as well as in MindSpore and
    save converted checkpoint with names,
    corresponds to inited DarkNet model.

    Args:
        cfg: Config parameters.

    Note:
        Convert weights without last FC layer.
    """
    darknet53 = DarkNet(
        ResidualBlock,
        cfg.backbone_layers,
        cfg.backbone_input_shape,
        cfg.backbone_shape,
        detect=True,
    )

    # Get MindSpore names of parameters
    ms_keys = list(darknet53.parameters_dict().keys())

    # Get PyTorch weights and names
    pt_weights = torch.load(cfg.ckpt_url, map_location=torch.device('cpu'))['state_dict']
    pt_keys = list(pt_weights.keys())

    # Remove redundant keys
    pt_keys_clear = [
        key
        for key in pt_keys
        if not key.endswith('tracked')
    ]

    # One layer consist of 5 parameters
    # Arrange PyTorch keys as well as in MindSpore
    pt_keys_aligned = []
    for block_num in range(len(pt_keys_clear[:-2]) // 5):
        layer = pt_keys_clear[block_num * 5:(block_num + 1) * 5]
        pt_keys_aligned.append(layer[0])  # Conv weight
        pt_keys_aligned.append(layer[3])  # BN moving mean
        pt_keys_aligned.append(layer[4])  # BN moving var
        pt_keys_aligned.append(layer[1])  # BN gamma
        pt_keys_aligned.append(layer[2])  # BN beta

    ms_checkpoint = []
    for key_ms, key_pt in zip(ms_keys, pt_keys_aligned):
        weight = Parameter(Tensor(pt_weights[key_pt].numpy(), mstype.float32))
        ms_checkpoint.append({'name': key_ms, 'data': weight})

    checkpoint_name = str(Path(cfg.ckpt_url).resolve().parent / 'darknet53.ckpt')
    save_checkpoint(ms_checkpoint, checkpoint_name)

    print(f'Checkpoint converted successfully! Location {checkpoint_name}')


if __name__ == '__main__':
    if not Path(config.ckpt_url).exists():
        raise FileNotFoundError(f'Expect a path to the PyTorch checkpoint, but not found it at "{config.ckpt_url}"')

    convert(config)

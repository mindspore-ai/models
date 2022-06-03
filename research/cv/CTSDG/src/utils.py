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
"""utils"""
from pathlib import Path

from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops


def check_args(cfg):
    """check args"""
    if cfg.device_target != 'GPU':
        raise ValueError(f'Only GPU device is supported now, got {cfg.device_target}')

    if cfg.file_format and cfg.file_format != 'MINDIR':
        raise ValueError(f'Only MINDIR format is supported for export now, got {cfg.file_format}')

    if not Path(cfg.checkpoint_path).exists():
        raise FileExistsError(f'checkpoint_path {cfg.checkpoint_path} doesn`t exist.')

    if not Path(cfg.pretrained_vgg).exists():
        raise FileExistsError(f'pretrained vgg feature extractor {cfg.pretrained_vgg} '
                              f'doesn`t exist.')

    if not isinstance(cfg.image_load_size, (tuple, list)):
        raise ValueError(f'config.image_load_size must be a list or a tuple!, '
                         f'got {type(cfg.image_load_size)}')


def extract_patches(inp, ksize, stride=1, pad=1, dilation=1):
    """unfold function"""
    batch_num, channel, height, width = inp.shape
    out_h = (height + pad + pad - ksize - (ksize - 1) * (dilation - 1)) // stride + 1
    out_w = (width + pad + pad - ksize - (ksize - 1) * (dilation - 1)) // stride + 1

    inp = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))(inp)
    res = ops.Zeros()((batch_num, channel, ksize, ksize, out_h, out_w), mstype.float32)

    for y in range(ksize):
        y_max = y + stride * out_h
        for x in range(ksize):
            x_max = x + stride * out_w
            res[:, :, y, x, :, :] = inp[:, :, y:y_max:stride, x:x_max:stride]

    res = res.transpose(0, 4, 5, 1, 2, 3)
    return res

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

"""Utils for model"""

import os
from os import path as osp
import random
import math
import cv2
import numpy as np
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.common import initializer as init


def init_weights_blocks(net, init_type='normal', init_gain=0.1):
    """
    Initialize blocks weights
    """
    for cell in net:
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            elif init_type == 'kaiming':
                cell.weight.set_data(init.initializer(init.HeNormal(init_gain), cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


def init_weights_network(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights
    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            elif init_type == 'kaiming':
                cell.weight.set_data(init.initializer(init.HeNormal(init_gain), cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


def CosineAnnealingRestartLR(period, restart_weights=(1.), eta_min=0, total_step=1):
    """ Cosine annealing with restarts learning rate scheme."""
    base_lr = 0.0002
    lr = []
    for i in range(total_step):
        now_period = i // period
        lr.append(eta_min + restart_weights[now_period] * 0.5 * (base_lr - eta_min) *
                  (1 + math.cos(math.pi * ((i % period) / period))))
    return lr


def calculate_psnr(img1, img2, crop_border=4, input_order='HWC', test_y_channel=True):
    """calculate psnr"""
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays."""
    result = []
    for _tensor in tensor:
        _tensor = C.clip_by_value(_tensor, *min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 3:
            img_np = _tensor.asnumpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file."""
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order."""
    if input_order not in ['HWC', 'CHW']:
        print('Wrong input_order. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def _convert_input_type_range(img):
    """Convert the type and range of the input image."""
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        print('Error,The img type should be np.float32 or np.uint8, ')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type."""
    if dst_type not in (np.uint8, np.float32):
        print('Error,The dst_type should be np.float32 or np.uint8')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image."""
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def to_y_channel(img):
    """Change to Y channel of YCbCr."""
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def paired_paths_from_folder(lq_folder, gt_folder):
    """Generate paired paths from folders."""
    gt_paths = list(os.listdir(gt_folder))
    paths = []
    for gt_path in gt_paths:
        input_path = osp.join(lq_folder, gt_path)
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([('lq_path', input_path), ('gt_path', gt_path)]))
    return paths


def get(filepath):
    """Get values according to the filepath """
    filepath = str(filepath)
    with open(filepath, 'rb') as f:
        value_buf = f.read()
    return value_buf


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes."""
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path):
    """Paired random crop. Support Numpy array and Tensor inputs."""
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        print('Error,h_gt != h_lq * scale or w_gt != w_lq * scale')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        print('Error,h_lq < lq_patch_size or w_lq < lq_patch_size')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def augment(imgs, hflip=False, rotation=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees)."""
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs


def img2tensor(imgs, bgr2rgb=True):
    """Numpy array to tensor."""

    def _totensor(img, bgr2rgb):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb) for img in imgs]

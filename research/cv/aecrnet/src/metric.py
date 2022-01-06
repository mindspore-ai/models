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
"""Metric for evaluation."""
import os
import math

import cv2
from PIL import Image
import numpy as np
from mindspore import nn, Tensor, ops
from mindspore import dtype as mstype
from mindspore.ops.operations.comm_ops import ReduceOp

try:
    from model_utils.device_adapter import get_rank_id, get_device_num
except ImportError:
    get_rank_id = None
    get_device_num = None
finally:
    pass

def quantize(img, rgb_range):
    """quantize image range to 0-255"""
    pixel_range = 255 / rgb_range
    img = np.multiply(img, pixel_range)
    img = np.clip(img, 0, 255)
    img = np.round(img) / pixel_range
    return img


def calc_psnr(sr, hr, scale, rgb_range):
    """calculate psnr"""
    hr = np.float32(hr)
    sr = np.float32(sr)
    diff = (sr - hr) / rgb_range
    gray_coeffs = np.array([65.738, 129.057, 25.064]).reshape((1, 3, 1, 1)) / 256
    diff = np.multiply(diff, gray_coeffs).sum(1)
    if hr.size == 1:
        return 0

    shave = scale
    valid = diff[..., shave:-shave, shave:-shave]
    mse = np.mean(pow(valid, 2))
    return -10 * math.log10(mse)


def rgb2ycbcr(img, y_only=True):
    """from rgb space to ycbcr space"""
    img.astype(np.float32)
    if y_only:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return rlt


def calc_ssim(img1, img2, scale):
    """calculate ssim"""
    def ssim(img1, img2):
        """calculate ssim"""
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) *
                                                                (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()

    border = scale
    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    if img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for _ in range(3):
                ssims.append(ssim(img1, img2))

            return np.array(ssims).mean()
        if img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))

    raise ValueError('Wrong input image dimensions.')

class TensorSyncer(nn.Cell):
    """
    sync metric values from all mindspore-processes
    """
    def __init__(self, t_type="sum"):
        super(TensorSyncer, self).__init__()
        self.t_type = t_type.lower()
        if self.t_type == "sum":
            self.ops = ops.AllReduce(ReduceOp.SUM)
        elif self.t_type == "gather":
            self.ops = ops.AllGather()
        else:
            raise ValueError(f"TensorSyncer.t_type == {self.t_type} is not support")

    def construct(self, x):
        return self.ops(x)


class _DistMetric(nn.Metric):
    """
    gather data from all rank while eval(True)
    _type(str): choice from ["avg", "sum"].
    """
    def __init__(self, t_type):
        super(_DistMetric, self).__init__()
        self.t_type = t_type.lower()
        self.all_reduce_sum = None
        if get_device_num is not None and get_device_num() > 1:
            self.all_reduce_sum = TensorSyncer(t_type="sum")
        self.clear()

    def _accumulate(self, value):
        if isinstance(value, (list, tuple)):
            self._acc_value += sum(value)
            self._count += len(value)
        else:
            self._acc_value += value
            self._count += 1

    def clear(self):
        self._acc_value = 0.0
        self._count = 0

    def eval(self, sync=True):
        """
        sync: True, return metric value merged from all mindspore-processes
        sync: False, return metric value in this single mindspore-processes
        """
        if self._count == 0:
            raise RuntimeError('self._count == 0')
        if self.sum is not None and sync:
            data = Tensor([self._acc_value, self._count], mstype.float32)
            data = self.all_reduce_sum(data)
            acc_value, count = self._convert_data(data).tolist()
        else:
            acc_value, count = self._acc_value, self._count
        if self.t_type == "avg":
            return acc_value / count
        if self.t_type == "sum":
            return acc_value
        raise RuntimeError(f"_DistMetric.t_type={self.t_type} is not support")


class PSNR(_DistMetric):
    """
    Define PSNR metric for SR network.
    """
    def __init__(self, rgb_range, shave):
        super(PSNR, self).__init__(t_type="avg")
        self.shave = shave
        self.rgb_range = rgb_range
        self.quantize = Quantizer(0.0, 255.0)

    def update(self, *inputs):
        """
        update psnr
        """
        if len(inputs) != 2:
            raise ValueError('PSNR need 2 inputs (sr, hr), but got {}'.format(len(inputs)))
        sr, hr = inputs
        sr = self.quantize(sr)
        diff = (sr - hr) / self.rgb_range
        valid = diff
        if self.shave is not None and self.shave != 0:
            valid = valid[..., self.shave:(-self.shave), self.shave:(-self.shave)]
        mse_list = (valid ** 2).mean(axis=(1, 2, 3))
        mse_list = self._convert_data(mse_list).tolist()
        psnr_list = [float(1e32) if mse == 0 else(- 10.0 * math.log10(mse)) for mse in mse_list]
        self._accumulate(psnr_list)

class Quantizer(nn.Cell):
    """
    clip by [0.0, 255.0], rount to int
    """
    def __init__(self, min_v=0.0, max_v=255.0):
        super(Quantizer, self).__init__()
        self.round = ops.Round()
        self.min_v = min_v
        self.max_v = max_v

    def construct(self, x):
        x = ops.clip_by_value(x, self.min_v, self.max_v)
        x = self.round(x)
        return x


class SaveSrHr(_DistMetric):
    """
    help to save sr and hr
    """
    def __init__(self, save_dir):
        super(SaveSrHr, self).__init__(t_type="sum")
        self.save_dir = save_dir
        self.quantize = Quantizer(0.0, 255.0)
        self.rank_id = 0 if get_rank_id is None else get_rank_id()
        self.device_num = 1 if get_device_num is None else get_device_num()

    def update(self, *inputs):
        """
        update images to save
        """
        if len(inputs) != 2:
            raise ValueError('SaveSrHr need 2 inputs (sr, hr), but got {}'.format(len(inputs)))
        sr, hr = inputs
        sr = self.quantize(sr)
        sr = self._convert_data(sr).astype(np.uint8)
        hr = self._convert_data(hr).astype(np.uint8)
        for s, h in zip(sr.transpose(0, 2, 3, 1), hr.transpose(0, 2, 3, 1)):
            idx = self._count * self.device_num + self.rank_id
            sr_path = os.path.join(self.save_dir, f"{idx:0>4}_sr.png")
            Image.fromarray(s).save(sr_path)
            hr_path = os.path.join(self.save_dir, f"{idx:0>4}_hr.png")
            Image.fromarray(h).save(hr_path)
            self._accumulate(1)


def convert_data(data):
    """
    Convert data type to numpy array.

    Args:
        data (Object): Input data.

    Returns:
        Ndarray, data with `np.ndarray` type.
    """
    if isinstance(data, Tensor):
        data = data.asnumpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError('The input data type must be a tensor, list or numpy.ndarray')
    return data

def save_srhr(save_dir, filename, *inputs):
    """save sr hr"""
    quantizer = Quantizer(0.0, 255.0)
    sr, hr = inputs
    sr = quantizer(sr)
    sr = convert_data(sr).astype(np.uint8)
    hr = convert_data(hr).astype(np.uint8)
    for s, h in zip(sr.transpose(0, 2, 3, 1), hr.transpose(0, 2, 3, 1)):
        sr_path = os.path.join(save_dir, f"{filename}_pred.png")
        Image.fromarray(s).save(sr_path)
        hr_path = os.path.join(save_dir, f"{filename}_gt.png")
        Image.fromarray(h).save(hr_path)

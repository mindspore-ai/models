# Copyright 2023 Huawei Technologies Co., Ltd
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

import logging
import cv2
import numpy as np
from scipy import signal


def logger_info(logger_name, log_path="default_logger.log"):
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        log.propagate = False
    print("LogHandlers setup!")
    level = logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d : %(message)s", datefmt="%y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path, mode="w")  # every log as new
    fh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.addHandler(sh)


def _rgb2ycbcr(img, max_val=255):
    ov = np.array([[16], [128], [128]])
    tm = np.array(
        [
            [0.256788235294118, 0.504129411764706, 0.097905882352941],
            [-0.148223529411765, -0.290992156862745, 0.439215686274510],
            [0.439215686274510, -0.367788235294118, -0.071427450980392],
        ]
    )

    if max_val == 1:
        ov = ov / 255.0

    t = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(tm))
    t[:, 0] += ov[0]
    t[:, 1] += ov[1]
    t[:, 2] += ov[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])
    return ycbcr


def cal_psnr(y_true, y_pred, shave_border=4):
    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return 20 * np.log10(255.0 / rmse)


def cal_ssim(img1, img2):
    k = [0.01, 0.03]
    l = 255
    kernel_x = cv2.getGaussianKernel(11, 1.5)
    window = kernel_x * kernel_x.T

    c1 = (k[0] * l) ** 2
    c2 = (k[1] * l) ** 2
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mu1 = signal.convolve2d(img1, window, "valid")
    mu2 = signal.convolve2d(img2, window, "valid")

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, "valid") - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, "valid") - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, "valid") - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    mssim = np.mean(ssim_map)
    return mssim

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
"""Utils scripts."""
import cv2
import numpy as np
from mindspore import numpy as msnp
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from skimage.measure import regionprops


def compute_sad_loss(pd, gt, mask):
    """
    Compute the SAD error given a prediction, a ground truth and a mask.

    Args:
        pd (np.array): Predicted alpha mask.
        gt (np.array): Groundtruth alpha mask.
        mask (np.array): Unknown region of trimap mask.

    Returns:
        loss (float): Computed SAD loss.
    """
    cv2.normalize(pd, pd, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.normalize(gt, gt, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    error_map = np.abs(pd - gt) / 255.
    loss = np.sum(error_map * mask)
    # the loss is scaled by 1000 due to the large images
    loss = loss / 1000
    return loss


def compute_mse_loss(pd, gt, mask):
    """
    Compute the MSE error.

    Args:
        pd (np.array): Predicted alpha mask.
        gt (np.array): Groundtruth alpha mask.
        mask (np.array): Unknown region of trimap mask.

    Returns:
        loss (float): Computed MSE loss.
    """
    cv2.normalize(pd, pd, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.normalize(gt, gt, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    error_map = (pd - gt) / 255.
    loss = np.sum(np.square(error_map) * mask) / np.sum(mask)
    return loss


def compute_gradient_loss(pd, gt, mask):
    """
    Compute the gradient error.

    Args:
        pd (np.array): Predicted alpha mask.
        gt (np.array): Groundtruth alpha mask.
        mask (np.array): Unknown region of trimap mask.

    Returns:
        loss (float): Computed Grad loss.
    """
    cv2.normalize(pd, pd, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.normalize(gt, gt, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    pd = pd / 255.
    gt = gt / 255.
    pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x ** 2 + pd_y ** 2)
    gt_mag = np.sqrt(gt_x ** 2 + gt_y ** 2)

    error_map = np.square(pd_mag - gt_mag)
    loss = np.sum(error_map * mask) / 10
    return loss


def compute_connectivity_loss(pd, gt, mask, step=0.1):
    """
    Compute the connectivity error.

    Args:
        pd (np.array): Predicted alpha mask.
        gt (np.array): Groundtruth alpha mask.
        mask (np.array): Unknown region of trimap mask.
        step (float): Threshold steps.

    Returns:
        loss (float): Computed Conn loss.
    """
    cv2.normalize(pd, pd, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(gt, gt, 0, 255, cv2.NORM_MINMAX)
    pd = pd / 255.
    gt = gt / 255.

    h, w = pd.shape

    thresh_steps = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pd >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]

        label_image = label(pd_th & gt_th, connectivity=1)
        cc = regionprops(label_image)
        size_vec = np.array([c.area for c in cc])
        if size_vec.size == 0:
            continue
        max_id = np.argmax(size_vec)
        coords = cc[max_id].coords

        omega = np.zeros((h, w), dtype=np.float32)
        omega[coords[:, 0], coords[:, 1]] = 1

        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    # the definition of lambda is ambiguous
    d_pd = pd - l_map
    d_gt = gt - l_map
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    loss = np.sum(np.abs(phi_pd - phi_gt) * mask) / 1000
    return loss


def image_alignment(x, output_stride, odd=False):
    """
    Resize inputs corresponds to stride.

    Args:
        x: Raw model inputs.
        output_stride: Output image stride.
        odd: Odd of the inputs size.

    Returns:
        new_x: Resized inputs.
    """
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv2.resize(x1, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    new_x2 = cv2.resize(x2, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)
    return new_x


def image_rescale(x, scale):
    """
    Rescale inputs.

    Args:
        x (dict): Raw model input.
        scale (float): Scale ratio.

    Returns:
        new_x (dict): Resized inputs.
    """
    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv2.resize(x1, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    new_x2 = cv2.resize(x2, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)
    return new_x


def compute_loss(diff, diff_mask, bs, epsilon=1e-6):
    """
    Compute MRSE loss with mask.

    Args:
        diff (np.array):
        diff_mask (np.array):
        bs (int): Batch size of array.
        epsilon (float): Additive value to except division by zero error.

    Returns:
        loss: Computed MRSE loss.
    """
    loss = msnp.sqrt(diff * diff + epsilon ** 2)
    loss = loss.sum(axis=2).sum(axis=2) / diff_mask.sum(axis=2).sum(axis=2)
    loss = loss.sum() / bs
    return loss


def weighted_loss(pd, mask, alpha, fg, bg, c_g, wl=0.5):
    """
    Compute weighted loss of alpha prediction and the composition during training.

    Args:
        pd: Input 4 channel image.
        mask: Image mask.
        alpha: Image trimap.
        fg: Original foreground image.
        bg: Original background image.
        c_g: Merged by mask foreground over background.
        wl (float): Loss weight.

    Returns:
        output: Weighted loss.
    """
    bs, _, h, w = pd.shape
    mask = mask.reshape((bs, 1, h, w))
    alpha_gt = alpha.reshape((bs, 1, h, w))
    diff_alpha = (pd - alpha_gt) * mask

    c_p = pd * fg + (1 - pd) * bg
    diff_color = (c_p - c_g) * mask

    loss_alpha = compute_loss(diff_alpha, mask, bs)
    loss_composition = compute_loss(diff_color, mask, bs)

    return wl * loss_alpha + (1 - wl) * loss_composition

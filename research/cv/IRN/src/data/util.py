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
""" Utility function for data processing."""

import os
import math
import random
import numpy as np
import cv2

####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_image_paths(data_type, dataroot):
    '''get image path list
    support image files'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(data_type))
    return paths, sizes


###################### read images ######################
def read_img(path, size=None):
    '''read image by cv2
    return: Numpy float32, HWC, BGR, [0,1]'''
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


####################
# image processing
# process on numpy image
####################


def augment(img_list, hflip=True, rot=True):
    '''image augment'''
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def augment_flow(img_list, flow_list, hflip=True, rot=True):
    '''image augment with flow'''
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            flow = flow[:, ::-1, :]
            flow[:, :, 0] *= -1
        if vflip:
            flow = flow[::-1, :, :]
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    rlt_img_list = [_augment(img) for img in img_list]
    rlt_flow_list = [_augment_flow(flow) for flow in flow_list]

    return rlt_img_list, rlt_flow_list


def channel_convert(in_c, tar_type, img_list):
    '''conversion among BGR, gray and y'''
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]

    if in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]

    if in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    return img_list


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    '''image crop'''
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, _ = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


####################
# Functions
####################


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = np.absolute(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).astype(absx.dtype)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).astype(absx.dtype))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    '''calculate weights indices'''
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = np.linspace(1, out_length, out_length).astype(np.float32)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = np.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices_1 = left.reshape(out_length, 1)
    indices_1 = np.tile(indices_1, (1, P))
    indices_2 = np.linspace(0, P - 1, P).reshape(1, P).astype(np.float32)
    indices_2 = np.tile(indices_2, (out_length, 1))
    indices = indices_1 + indices_2

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.reshape(out_length, 1)
    distance_to_center = np.tile(distance_to_center, (1, P)) - indices

    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = np.sum(weights, 1).reshape(out_length, 1)
    weights_sum = np.tile(weights_sum, (1, P))
    weights = weights / weights_sum

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = np.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices[:, 1:(P - 1)]
        weights = weights[:, 1:(P - 1)]
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices[:, 0:(P - 2)]
        weights = weights[:, 0:(P - 2)]

    weights = np.ascontiguousarray(weights)
    indices = np.ascontiguousarray(indices)
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    '''Now the scale should be the same for H and W
    input: img: CHW RGB [0,1]
    output: CHW RGB [0,1] w/o round'''

    in_C, in_H, in_W = img.shape
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying

    # img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    # img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)
    img_aug = np.ones(
        (in_C, in_H + sym_len_Hs + sym_len_He, in_W)).astype(np.float32)
    img_aug[:, sym_len_Hs:sym_len_Hs+in_H] = img.copy()

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = np.arange(sym_patch.shape[1]-1, -1, -1).astype(np.int64)
    sym_patch_inv = sym_patch[:, inv_idx]
    img_aug[:, 0:sym_len_Hs] = sym_patch_inv.copy()

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = sym_patch[:, inv_idx]
    img_aug[:, sym_len_Hs + in_H:sym_len_Hs +
            in_H + sym_len_He] = sym_patch_inv.copy()

    out_1 = np.ones((in_C, out_H, in_W)).astype(np.float32)
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = np.matmul(
            img_aug[0, idx:idx + kernel_width, :].transpose(1, 0), weights_H[i])
        out_1[1, i, :] = np.matmul(
            img_aug[1, idx:idx + kernel_width, :].transpose(1, 0), weights_H[i])
        out_1[2, i, :] = np.matmul(
            img_aug[2, idx:idx + kernel_width, :].transpose(1, 0), weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = np.ones((in_C, out_H, in_W + sym_len_Ws +
                         sym_len_We)).astype(np.float32)
    out_1_aug[:, :, sym_len_Ws:sym_len_Ws+in_W] = out_1.copy()

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = np.arange(sym_patch.shape[2] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = sym_patch[:, :, inv_idx]
    out_1_aug[:, :, 0:sym_len_Ws] = sym_patch_inv.copy()

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = np.arange(sym_patch.shape[2] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = sym_patch[:, :, inv_idx]
    out_1_aug[:, :, sym_len_Ws + in_W: sym_len_Ws +
              in_W + sym_len_We] = sym_patch_inv.copy()

    out_2 = np.ones((in_C, out_H, out_W))
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = np.matmul(
            out_1_aug[0, :, idx:idx + kernel_width], weights_W[i])
        out_2[1, :, i] = np.matmul(
            out_1_aug[1, :, idx:idx + kernel_width], weights_W[i])
        out_2[2, :, i] = np.matmul(
            out_1_aug[2, :, idx:idx + kernel_width], weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True):
    '''Now the scale should be the same for H and W
    input: img: Numpy, HWC BGR [0,1]
    output: HWC BGR [0,1] w/o round'''

    in_H, in_W, in_C = img.shape
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)

    # process H dimension
    # symmetric copying
    img_aug = np.ones((in_H + sym_len_Hs + sym_len_He,
                       in_W, in_C)).astype(np.float32)
    img_aug[sym_len_Hs:sym_len_Hs+in_H] = img.copy()
    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = np.arange(sym_patch.shape[0] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = sym_patch[inv_idx]
    img_aug[0:sym_len_Hs] = sym_patch_inv.copy()

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = np.arange(sym_patch.shape[0] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = sym_patch[inv_idx]
    img_aug[sym_len_Hs + in_H: sym_len_Hs +
            in_H + sym_len_He] = sym_patch_inv.copy()

    out_1 = np.ones((out_H, in_W, in_C)).astype(np.float32)
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = np.matmul(
            img_aug[idx:idx + kernel_width, :, 0].transpose(1, 0), weights_H[i])
        out_1[i, :, 1] = np.matmul(
            img_aug[idx:idx + kernel_width, :, 1].transpose(1, 0), weights_H[i])
        out_1[i, :, 2] = np.matmul(
            img_aug[idx:idx + kernel_width, :, 2].transpose(1, 0), weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = np.ones(
        (out_H, in_W + sym_len_Ws + sym_len_We, in_C)).astype(np.float32)
    out_1_aug[:, sym_len_Ws:sym_len_Ws+in_W] = out_1.copy()

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = sym_patch[:, inv_idx]
    out_1_aug[:, 0:sym_len_Ws] = sym_patch_inv.copy()

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype(np.int64)
    sym_patch_inv = sym_patch[:, inv_idx]
    out_1_aug[:, sym_len_Ws + in_W:sym_len_Ws +
              in_W + sym_len_We] = sym_patch_inv.copy()

    out_2 = np.ones((out_H, out_W, in_C)).astype(np.float32)
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = np.matmul(
            out_1_aug[:, idx:idx + kernel_width, 0], weights_W[i])
        out_2[:, i, 1] = np.matmul(
            out_1_aug[:, idx:idx + kernel_width, 1], weights_W[i])
        out_2[:, i, 2] = np.matmul(
            out_1_aug[:, idx:idx + kernel_width, 2], weights_W[i])

    return out_2
